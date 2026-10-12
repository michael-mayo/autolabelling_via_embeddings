"""YOLOE visual-prompt detection as an `ImageProber`.

YOLOE (Wang et al., 2025; Ultralytics) is a real-time open-set YOLO detector
whose classification head scores every anchor against a set of class
embeddings. Its SAVPE module turns a visual prompt, a mask over an image's
neck features, into such an embedding, so a prompt can come from one image
and be searched for in others.

- probe embedding: the probe is resized so its longer side is `probe_size`
  and placed at the top-left of a grey `imgsz` canvas, and its area is the
  prompt mask. The backbone and neck run once per batch, then SAVPE gives one
  512-d embedding per probe.
- image embedding: the image is letterboxed to `imgsz` (aspect kept, grey
  padding at the bottom and right), and the three neck feature maps the head
  reads are kept.
- score: the detection head (without the mask branch) runs once per image,
  with every probe's embedding as a class, giving each anchor a box and a
  sigmoid score per probe. A probe's probability is its best anchor score.
  Its boxes are the anchors scoring at least `box_threshold`, after NMS.

Weights are Ultralytics release assets (AGPL-3.0), downloaded once to
`<weights_dir>/ultralytics/`.
"""

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from ultralytics import YOLOE
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.utils.downloads import attempt_download_asset

from .image_prober import ImageProber
from .image_prober import ProbeResult
from .image_prober import run_cli
from .image_prober import select_boxes

# Letterbox padding value used by Ultralytics.
PAD_GREY = (114, 114, 114)
# Stride of the visual-prompt mask relative to the input (SAVPE reads 1/8-scale masks).
PROMPT_STRIDE = 8


class ImageProber_YOLOE(ImageProber):
    """Probe images with YOLOE, using each probe crop as a visual prompt."""

    config_section = "image_prober_yoloe"

    def __init__(
        self,
        weights_file: str = "yoloe-11l-seg.pt",
        dtype: str = "float16",
        imgsz: int = 640,
        probe_size: int = 160,
        box_threshold: float = 0.05,
        nms_iou: float = 0.5,
        max_boxes: int = 20,
        **common: Any,
    ) -> None:
        """Load YOLOE from the weights folder (downloading once); `common` goes to `ImageProber`.

        `weights_file` is an Ultralytics YOLOE asset with a prompt-able
        (not prompt-free, `-pf`) head that uses NMS, e.g. `yoloe-11s/m/l-seg.pt`.
        """
        super().__init__(**common)
        if imgsz % 32 or probe_size > imgsz:
            raise ValueError("imgsz must be a multiple of 32 and probe_size at most imgsz")
        self.weights_file = weights_file
        self.imgsz = imgsz
        self.probe_size = probe_size
        self.box_threshold = box_threshold
        self.nms_iou = nms_iou
        self.max_boxes = max_boxes
        path = self._ensure_asset(weights_file)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.net = YOLOE(str(path)).model.to(self.device, self.dtype).eval()
        self.head = self.net.model[-1]
        if not isinstance(self.head, YOLOEDetect) or self.head.end2end or hasattr(self.head, "lrpc"):
            raise ValueError(f"{weights_file} needs a prompt-able YOLOE head with NMS (e.g. yoloe-11l-seg.pt)")

    def _ensure_asset(self, name: str) -> Path:
        """Return the local path of an Ultralytics release asset, downloading it only if missing."""
        path = self.weights_dir / "ultralytics" / name
        if path.exists():
            print(f"[{type(self).__name__}] using weights in {path}")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{type(self).__name__}] downloading {name} to {path}")
        return Path(attempt_download_asset(str(path)))

    @property
    def cache_tag(self) -> str:
        """Model name plus the input and probe sizes, which change the embeddings."""
        return f"yoloe-{Path(self.weights_file).stem}-i{self.imgsz}-p{self.probe_size}"

    def _to_tensor(self, canvas: Image.Image) -> torch.Tensor:
        """Convert an RGB canvas to a [3, H, W] float tensor in [0, 1]."""
        data = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        return data.view(canvas.height, canvas.width, 3).permute(2, 0, 1).float().div(255)

    def _letterbox(self, image: Image.Image, long_side: int) -> tuple[torch.Tensor, float, tuple[int, int]]:
        """Resize `image` so its longer side is `long_side` and paste it top-left on a grey `imgsz` square.

        Returns the canvas tensor, the resize factor and the resized (w, h).
        """
        scale = long_side / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        canvas = Image.new("RGB", (self.imgsz, self.imgsz), PAD_GREY)
        canvas.paste(image.resize(size, Image.Resampling.BILINEAR), (0, 0))
        return self._to_tensor(canvas), scale, size

    def _neck(self, pixels: torch.Tensor) -> list[torch.Tensor]:
        """Run the backbone and neck once; return the feature maps the detection head reads."""
        outputs: list[torch.Tensor | None] = []
        x: Any = pixels.to(self.device, self.dtype)
        for module in self.net.model[:-1]:
            if module.f != -1:
                x = outputs[module.f] if isinstance(module.f, int) else [x if j == -1 else outputs[j] for j in module.f]
            x = module(x)
            outputs.append(x if module.i in self.net.save else None)
        self.embed_passes += 1
        return [outputs[j] for j in self.head.f]

    def _embed_images(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """Return each image's neck feature maps and the factor mapping its pixels to the letterbox."""
        boxed = [self._letterbox(image, self.imgsz) for image in images]
        feats = self._neck(torch.stack([t for t, _, _ in boxed]))
        return [{"feats": [f[i].cpu() for f in feats], "scale": scale} for i, (_, scale, _) in enumerate(boxed)]

    def _embed_probes(self, probes: list[Image.Image]) -> list[torch.Tensor]:
        """Return one SAVPE embedding per probe, prompted by a mask over the probe's area."""
        boxed = [self._letterbox(probe, self.probe_size) for probe in probes]
        feats = self._neck(torch.stack([t for t, _, _ in boxed]))
        grid = self.imgsz // PROMPT_STRIDE
        masks = torch.zeros(len(probes), 1, grid, grid, device=self.device, dtype=self.dtype)
        for i, (_, _, (w, h)) in enumerate(boxed):
            masks[i, 0, :max(1, round(h / PROMPT_STRIDE)), :max(1, round(w / PROMPT_STRIDE))] = 1
        embeddings = self.head.get_vpe(feats, masks)  # [B, 1, 512]
        return [e[0].cpu() for e in embeddings]

    def _score(self, image_embedding: dict[str, Any], probe_embeddings: list[torch.Tensor]) -> list[ProbeResult]:
        """Run the detection head once on one image with every probe embedding as a class."""
        feats = [f[None].to(self.device, self.dtype) for f in image_embedding["feats"]]
        classes = torch.stack(probe_embeddings)[None].to(self.device, self.dtype)  # [1, M, 512]
        self.head.nc = len(probe_embeddings)
        with torch.inference_mode():
            # YOLOEDetect.forward skips the segmentation head's mask prototypes, which are not needed.
            decoded, _ = YOLOEDetect.forward(self.head, feats + [classes])
        self.score_passes += 1
        decoded = decoded[0].float()  # [4 + M (+ mask coefficients), anchors]
        cx, cy, w, h = decoded[:4] / image_embedding["scale"]
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
        scores = decoded[4:4 + len(probe_embeddings)]
        return [
            ProbeResult(
                probability=scores[j].max().item(),
                boxes=select_boxes(boxes, scores[j], self.box_threshold, self.nms_iou, self.max_boxes),
            )
            for j in range(len(probe_embeddings))
        ]


if __name__ == "__main__":
    run_cli(ImageProber_YOLOE)
