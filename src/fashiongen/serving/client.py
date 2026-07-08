"""
client.py
=========
Triton inference client. Talks to a running Triton server (which serves the
TensorRT engines described by the config.pbtxt files) to:
  * run SDXL UNet denoising steps remotely, and
  * embed images with the CLIP image tower for search.

Requires: tritonclient[http]. Start the server first, e.g.:

    tritonserver --model-repository=serving/model_repository

Model repository layout Triton expects:
    model_repository/
      sdxl_unet/    { config.pbtxt, 1/unet.plan }
      clip_image/   { config.pbtxt, 1/clip_image.plan }
"""
from __future__ import annotations

import numpy as np


class TritonClient:
    def __init__(self, url: str = "localhost:8000"):
        import tritonclient.http as httpclient
        self._http = httpclient
        self.client = httpclient.InferenceServerClient(url=url)

    def _infer(self, model: str, inputs: dict, output_names: list[str]):
        ins = []
        for name, arr in inputs.items():
            arr = np.ascontiguousarray(arr)
            dtype = "FP16" if arr.dtype == np.float16 else "FP32"
            t = self._http.InferInput(name, arr.shape, dtype)
            t.set_data_from_numpy(arr)
            ins.append(t)
        outs = [self._http.InferRequestedOutput(n) for n in output_names]
        res = self.client.infer(model_name=model, inputs=ins, outputs=outs)
        return {n: res.as_numpy(n) for n in output_names}

    def clip_image_embed(self, pixel_values: np.ndarray) -> np.ndarray:
        """pixel_values: (B, 3, 224, 224) fp16 -> (B, 768) L2-normalised."""
        out = self._infer("clip_image", {"pixel_values": pixel_values.astype(np.float16)},
                          ["image_embeds"])["image_embeds"].astype(np.float32)
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)

    def unet_step(self, sample, timestep, encoder_hidden_states,
                  text_embeds, time_ids) -> np.ndarray:
        return self._infer("sdxl_unet", {
            "sample": sample, "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embeds": text_embeds, "time_ids": time_ids,
        }, ["noise_pred"])["noise_pred"]

    def health(self) -> bool:
        return self.client.is_server_ready()
