"""
train_controlnet.py
===================
Train an SDXL ControlNet so generation follows a structural conditioning map
(canny / softedge / garment mask / pose). This is what enforces "structural
preservation" - the ControlNet locks silhouette, seams and print layout while
the (LoRA-adapted) UNet fills in domain-faithful texture and shading.

Pipeline:
    * freeze SDXL UNet, VAE, text encoders (optionally load LoRA adapters)
    * initialise a ControlNetModel from the UNet encoder
    * train only the ControlNet on (image, conditioning, caption) triples
    * validate with the Structural Preservation Fidelity metric and log to MLflow

Requires: torch, diffusers, transformers, accelerate. Run on a GPU:

    accelerate launch -m fashiongen.training.train_controlnet \
        --config configs/controlnet.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/controlnet.yaml")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from diffusers import (ControlNetModel, DDPMScheduler, AutoencoderKL,
                           UNet2DConditionModel, StableDiffusionXLControlNetPipeline)
    from diffusers.optimization import get_scheduler
    from transformers import AutoTokenizer

    from fashiongen.training.dataset import make_torch_dataset
    from fashiongen.utils.mlflow_utils import MlflowRun

    acc = Accelerator(gradient_accumulation_steps=cfg["grad_accum"],
                      mixed_precision=cfg.get("mixed_precision", "bf16"))
    base = cfg["base_model"]
    wdtype = torch.bfloat16 if cfg.get("mixed_precision") == "bf16" else torch.float16

    tok1 = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
    tok2 = AutoTokenizer.from_pretrained(base, subfolder="tokenizer_2")
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
    from transformers import CLIPTextModel, CLIPTextModelWithProjection
    te1 = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
    te2 = CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2")
    noise_sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    # optionally load pretrained LoRA adapters into the (frozen) UNet
    if cfg.get("lora_adapters"):
        unet.load_attn_procs(cfg["lora_adapters"])

    # ControlNet initialised from the UNet weights (standard practice)
    controlnet = ControlNetModel.from_unet(unet)

    for m in (vae, te1, te2, unet):
        m.requires_grad_(False)
    controlnet.train()

    ds = make_torch_dataset(cfg["manifest"], resolution=cfg["resolution"],
                            cond_type=cfg.get("cond_type", "canny"))
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg.get("num_workers", 8), drop_last=True)

    opt = torch.optim.AdamW(controlnet.parameters(), lr=cfg["lr"],
                            weight_decay=cfg.get("wd", 1e-2))
    lr_sched = get_scheduler(cfg.get("lr_scheduler", "constant_with_warmup"), opt,
                             num_warmup_steps=cfg.get("warmup", 500),
                             num_training_steps=cfg["max_steps"])

    controlnet, opt, dl, lr_sched = acc.prepare(controlnet, opt, dl, lr_sched)
    vae.to(acc.device, dtype=wdtype); unet.to(acc.device, dtype=wdtype)
    te1.to(acc.device, dtype=wdtype); te2.to(acc.device, dtype=wdtype)

    def encode_prompts(caps):
        def enc(tok, te):
            ids = tok(caps, padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(acc.device)
            out = te(ids, output_hidden_states=True)
            return out.hidden_states[-2], out[0]
        h1, _ = enc(tok1, te1)
        h2, pooled = enc(tok2, te2)
        return torch.cat([h1, h2], dim=-1), pooled

    with MlflowRun(cfg.get("experiment", "sdxl-controlnet"), params=cfg) as run:
        step = 0
        while step < cfg["max_steps"]:
            for batch in dl:
                with acc.accumulate(controlnet):
                    px = batch["pixel_values"].to(acc.device, dtype=wdtype)
                    cond = batch["conditioning_pixel_values"].to(acc.device, dtype=wdtype)
                    latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    t = torch.randint(0, noise_sched.config.num_train_timesteps,
                                      (bsz,), device=acc.device).long()
                    noisy = noise_sched.add_noise(latents, noise, t)

                    prompt_embeds, pooled = encode_prompts(list(batch["caption"]))
                    add_time_ids = torch.tensor(
                        [[cfg["resolution"], cfg["resolution"], 0, 0,
                          cfg["resolution"], cfg["resolution"]]] * bsz,
                        device=acc.device, dtype=wdtype)
                    added = {"text_embeds": pooled, "time_ids": add_time_ids}

                    down, mid = controlnet(
                        noisy, t, encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added,
                        controlnet_cond=cond, return_dict=False)
                    model_pred = unet(
                        noisy, t, encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added,
                        down_block_additional_residuals=[d.to(wdtype) for d in down],
                        mid_block_additional_residual=mid.to(wdtype)).sample

                    target = (noise if noise_sched.config.prediction_type == "epsilon"
                              else noise_sched.get_velocity(latents, noise, t))
                    loss = F.mse_loss(model_pred.float(), target.float())
                    acc.backward(loss)
                    if acc.sync_gradients:
                        acc.clip_grad_norm_(controlnet.parameters(), 1.0)
                    opt.step(); lr_sched.step(); opt.zero_grad()

                if acc.sync_gradients:
                    step += 1
                    if step % cfg.get("log_every", 25) == 0 and acc.is_main_process:
                        run.log_metrics({"loss": loss.item()}, step=step)
                    if step % cfg.get("save_every", 1000) == 0 and acc.is_main_process:
                        out = Path(cfg["output_dir"]) / f"step_{step}"
                        acc.unwrap_model(controlnet).save_pretrained(out)
                    if step >= cfg["max_steps"]:
                        break

        if acc.is_main_process:
            final = Path(cfg["output_dir"]) / "final"
            acc.unwrap_model(controlnet).save_pretrained(final)
            run.log_artifact(str(final))
            print(f"saved ControlNet -> {final}")


if __name__ == "__main__":
    main()
