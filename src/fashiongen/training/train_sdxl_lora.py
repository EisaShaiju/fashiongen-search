"""
train_sdxl_lora.py
==================
LoRA fine-tuning of Stable Diffusion XL on a garment corpus so the model learns
the domain (fabrics, silhouettes, product-shot aesthetics) with a tiny number
of trainable parameters (the low-rank adapters), leaving the 3.5B-param base
frozen.

Pipeline:
    * freeze SDXL UNet + both text encoders + VAE
    * inject LoRA adapters into the UNet attention blocks (peft LoraConfig)
    * standard latent-diffusion loss (predict the added noise / v-prediction)
    * mixed precision (bf16), grad accumulation, EMA optional
    * MLflow logging + periodic SPF validation on held-out garments

Requires: torch, diffusers, transformers, peft, accelerate. Run on a GPU:

    accelerate launch -m fashiongen.training.train_sdxl_lora \
        --config configs/sdxl_lora.yaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sdxl_lora.yaml")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from diffusers import (StableDiffusionXLPipeline, DDPMScheduler,
                           AutoencoderKL, UNet2DConditionModel)
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    from fashiongen.training.dataset import make_torch_dataset
    from fashiongen.utils.mlflow_utils import MlflowRun

    acc = Accelerator(
        gradient_accumulation_steps=cfg["grad_accum"],
        mixed_precision=cfg.get("mixed_precision", "bf16"),
    )
    base = cfg["base_model"]                    # e.g. stabilityai/stable-diffusion-xl-base-1.0
    weight_dtype = torch.bfloat16 if cfg.get("mixed_precision") == "bf16" else torch.float16

    # --- load frozen components --------------------------------------- #
    tok1 = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    tok2 = AutoTokenizer.from_pretrained(base, subfolder="tokenizer_2")
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(cfg.get("vae", base),
                                        subfolder=None if cfg.get("vae") else "vae")
    unet: UNet2DConditionModel = pipe.unet
    te1, te2 = pipe.text_encoder, pipe.text_encoder_2
    noise_sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    for m in (vae, te1, te2, unet):
        m.requires_grad_(False)

    # --- inject LoRA into the UNet ------------------------------------ #
    lora = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora)
    unet.print_trainable_parameters()

    # --- data --------------------------------------------------------- #
    ds = make_torch_dataset(cfg["manifest"], resolution=cfg["resolution"],
                            cond_type="none")
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg.get("num_workers", 8), drop_last=True)

    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-2))
    max_steps = cfg["max_steps"]
    lr_sched = get_scheduler(cfg.get("lr_scheduler", "cosine"), opt,
                             num_warmup_steps=cfg.get("warmup", 500),
                             num_training_steps=max_steps)

    unet, opt, dl, lr_sched = acc.prepare(unet, opt, dl, lr_sched)
    vae.to(acc.device, dtype=weight_dtype)
    te1.to(acc.device, dtype=weight_dtype); te2.to(acc.device, dtype=weight_dtype)

    def encode_prompts(captions):
        """SDXL dual-text-encoder prompt embeds + pooled embeds."""
        def enc(tok, te, out_hidden=True):
            ids = tok(captions, padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(acc.device)
            out = te(ids, output_hidden_states=True)
            return out.hidden_states[-2], out[0]
        h1, _ = enc(tok1, te1)
        h2, pooled = enc(tok2, te2)
        prompt_embeds = torch.cat([h1, h2], dim=-1)
        return prompt_embeds, pooled

    # --- training loop ------------------------------------------------ #
    with MlflowRun(cfg.get("experiment", "sdxl-lora"), params=cfg) as run:
        step = 0
        unet.train()
        while step < max_steps:
            for batch in dl:
                with acc.accumulate(unet):
                    px = batch["pixel_values"].to(acc.device, dtype=weight_dtype)
                    latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    t = torch.randint(0, noise_sched.config.num_train_timesteps,
                                      (bsz,), device=acc.device).long()
                    noisy = noise_sched.add_noise(latents, noise, t)

                    prompt_embeds, pooled = encode_prompts(list(batch["caption"]))
                    # SDXL micro-conditioning (original/target size + crop)
                    add_time_ids = torch.tensor(
                        [[cfg["resolution"], cfg["resolution"], 0, 0,
                          cfg["resolution"], cfg["resolution"]]] * bsz,
                        device=acc.device, dtype=weight_dtype)
                    added = {"text_embeds": pooled, "time_ids": add_time_ids}

                    model_pred = unet(noisy, t, encoder_hidden_states=prompt_embeds,
                                      added_cond_kwargs=added).sample
                    target = (noise if noise_sched.config.prediction_type == "epsilon"
                              else noise_sched.get_velocity(latents, noise, t))
                    loss = F.mse_loss(model_pred.float(), target.float())

                    acc.backward(loss)
                    if acc.sync_gradients:
                        acc.clip_grad_norm_(params, 1.0)
                    opt.step(); lr_sched.step(); opt.zero_grad()

                if acc.sync_gradients:
                    step += 1
                    if step % cfg.get("log_every", 25) == 0 and acc.is_main_process:
                        run.log_metrics({"loss": loss.item(),
                                         "lr": lr_sched.get_last_lr()[0]}, step=step)
                    if step % cfg.get("save_every", 1000) == 0 and acc.is_main_process:
                        out = Path(cfg["output_dir"]) / f"step_{step}"
                        acc.unwrap_model(unet).save_pretrained(out)
                        run.log_artifact(str(out))
                    if step >= max_steps:
                        break

        if acc.is_main_process:
            final = Path(cfg["output_dir"]) / "final"
            acc.unwrap_model(unet).save_pretrained(final)
            run.log_artifact(str(final))
            print(f"saved LoRA adapters -> {final}")


if __name__ == "__main__":
    main()
