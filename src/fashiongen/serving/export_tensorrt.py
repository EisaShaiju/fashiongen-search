"""
export_tensorrt.py
==================
Optimise serving by exporting the heavy modules to TensorRT engines. The SDXL
UNet dominates both latency and VRAM at inference; converting it (and the
ControlNet + CLIP image tower) to TensorRT with FP16 (optionally INT8) cuts
VRAM and speeds up each denoising step.

Flow:  torch module -> ONNX (dynamic axes) -> TensorRT engine (fp16/int8).

Requires: torch, onnx, tensorrt (and the target NVIDIA GPU). This produces the
`.plan` engines that Triton loads (see serving/triton/*/config.pbtxt).

    python -m fashiongen.serving.export_tensorrt \
        --module unet --base stabilityai/stable-diffusion-xl-base-1.0 \
        --lora artifacts/lora/final --fp16 --out serving/engines/unet.plan
"""
from __future__ import annotations

import argparse
from pathlib import Path


def export_onnx(module, dummy_inputs, input_names, output_names,
                dynamic_axes, onnx_path, opset=17):
    import torch
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module, dummy_inputs, onnx_path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True)
    print(f"ONNX -> {onnx_path}")


def build_engine(onnx_path, engine_path, fp16=True, int8=False,
                 workspace_gb=8, min_batch=1, opt_batch=2, max_batch=4):
    """Build a TensorRT engine from ONNX with an optimisation profile."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)      # needs a calibrator in practice

    # dynamic-shape optimisation profile (latent HxW for 1024px SDXL = 128x128)
    profile = builder.create_optimization_profile()
    for inp in [network.get_input(i) for i in range(network.num_inputs)]:
        shape = inp.shape
        if shape[0] == -1:                          # dynamic batch on axis 0
            lo = [min_batch] + list(shape[1:])
            opt = [opt_batch] + list(shape[1:])
            hi = [max_batch] + list(shape[1:])
            profile.set_shape(inp.name, lo, opt, hi)
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"TensorRT engine -> {engine_path}  (fp16={fp16}, int8={int8})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", choices=["unet", "controlnet", "clip_image"],
                    required=True)
    ap.add_argument("--base", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--controlnet", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()

    import torch
    onnx_path = str(Path(args.out).with_suffix(".onnx"))

    if args.module == "unet":
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(args.base, subfolder="unet").eval()
        if args.lora:
            unet.load_attn_procs(args.lora)
        # SDXL UNet latent inputs at 1024px -> 128x128x4 latents
        sample = torch.randn(1, 4, 128, 128)
        timestep = torch.tensor([1], dtype=torch.float32)
        enc = torch.randn(1, 77, 2048)
        text_embeds = torch.randn(1, 1280)
        time_ids = torch.randn(1, 6)
        dummies = (sample, timestep, enc,
                   {"text_embeds": text_embeds, "time_ids": time_ids})
        export_onnx(
            unet, dummies,
            input_names=["sample", "timestep", "encoder_hidden_states",
                         "text_embeds", "time_ids"],
            output_names=["noise_pred"],
            dynamic_axes={"sample": {0: "B"}, "encoder_hidden_states": {0: "B"},
                          "text_embeds": {0: "B"}, "time_ids": {0: "B"},
                          "noise_pred": {0: "B"}},
            onnx_path=onnx_path)

    elif args.module == "clip_image":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k")
        visual = model.visual.eval()
        dummies = (torch.randn(1, 3, 224, 224),)
        export_onnx(visual, dummies, ["pixel_values"], ["image_embeds"],
                    {"pixel_values": {0: "B"}, "image_embeds": {0: "B"}}, onnx_path)

    elif args.module == "controlnet":
        from diffusers import ControlNetModel
        cn = ControlNetModel.from_pretrained(args.controlnet).eval()
        sample = torch.randn(1, 4, 128, 128)
        dummies = (sample, torch.tensor([1.0]), torch.randn(1, 77, 2048),
                   torch.randn(1, 3, 1024, 1024))
        export_onnx(cn, dummies,
                    ["sample", "timestep", "encoder_hidden_states", "controlnet_cond"],
                    ["down_residuals", "mid_residual"],
                    {"sample": {0: "B"}, "controlnet_cond": {0: "B"}}, onnx_path)

    build_engine(onnx_path, args.out, fp16=args.fp16, int8=args.int8)


if __name__ == "__main__":
    main()
