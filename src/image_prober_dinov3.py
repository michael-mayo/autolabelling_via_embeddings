"""DINOv3 patch-feature matching as an `ImageProber`.

DINOv3 (Meta, 2025) is a self-supervised vision transformer whose patch
features are strong generic descriptors, with no notion of classes or
boxes. This prober is training-free:

- image embedding: the L2-normalised patch features of the image, resized
  so its longer side is `image_size` (aspect ratio kept);
- probe embedding: a prototype, the normalised mean of the probe's patch
  features after trimming a `probe_border` margin (probe crops carry some
  context around the object);
- score: cosine similarity of the prototype with every image patch. The
  presence score is the mean of the `top_k` best patches, calibrated as
  `sigmoid((score - sim_offset) / sim_temperature)`. Boxes are the
  bounding boxes of connected regions of patches whose calibrated
  similarity reaches `box_threshold`.

Each image and probe needs one backbone pass (images of equal shape share
a batch); scoring is a matrix product. Any Hugging Face DINO checkpoint
that `AutoModel` loads works; `ImageProber_DINOv2` (image_prober_dinov2.py)
reuses this class for the ungated DINOv2 models.
"""

from collections import defaultdict
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel

from .image_prober import DetectedBox
from .image_prober import ImageProber
from .image_prober import ProbeResult
from .image_prober import run_cli

# ImageNet normalisation used by DINOv2 and DINOv3's web-image (LVD) checkpoints.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageProber_DINOv3(ImageProber):
    """Probe images by cosine similarity between a probe prototype and image patch features."""

    config_section = "image_prober_dinov3"

    def __init__(
        self,
        repo_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        dtype: str = "float16",
        image_size: int = 1024,
        probe_size: int = 224,
        probe_border: float = 0.1,
        top_k: int = 3,
        sim_offset: float = 0.48,
        sim_temperature: float = 0.04,
        box_threshold: float = 0.5,
        max_boxes: int = 20,
        **common: Any,
    ) -> None:
        """Load the DINO backbone from the weights folder (downloading once); `common` goes to `ImageProber`.

        DINOv3 checkpoints are gated: accept the licence on the model page and
        log in (`hf auth login`) before the first download.
        """
        super().__init__(**common)
        self.repo_id = repo_id
        self.image_size = image_size
        self.probe_size = probe_size
        self.probe_border = probe_border
        self.top_k = top_k
        self.sim_offset = sim_offset
        self.sim_temperature = sim_temperature
        self.box_threshold = box_threshold
        self.max_boxes = max_boxes
        path = self.ensure_weights(repo_id, allow_patterns=["*.json", "*.safetensors"])
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(path, dtype=self.dtype).to(self.device).eval()
        config = self.model.config
        self.patch_size = config.patch_size
        # Tokens before the patch grid: the class token plus any register tokens.
        self.num_prefix = 1 + getattr(config, "num_register_tokens", 0)
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(3, 1, 1)

    @property
    def cache_tag(self) -> str:
        """Model name plus the input sizes and probe trimming, which change the embeddings."""
        return f"dino-{self.repo_id.split('/')[-1]}-i{self.image_size}-p{self.probe_size}-b{self.probe_border}"

    def _resize(self, image: Image.Image, long_side: int) -> Image.Image:
        """Resize so the longer side is about `long_side`, both sides a multiple of the patch size."""
        scale = long_side / max(image.size)
        w, h = (max(self.patch_size, round(v * scale / self.patch_size) * self.patch_size) for v in image.size)
        return image.resize((w, h), Image.Resampling.BICUBIC)

    def _patch_features(self, images: list[Image.Image]) -> list[torch.Tensor]:
        """Return normalised patch features [rows, cols, dim] per image, one pass per distinct shape."""
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, image in enumerate(images):
            groups[image.size].append(i)
        features: list[torch.Tensor | None] = [None] * len(images)
        for (w, h), indices in groups.items():
            pixels = torch.stack([
                torch.frombuffer(bytearray(images[i].tobytes()), dtype=torch.uint8).view(h, w, 3).permute(2, 0, 1)
                for i in indices
            ]).to(self.device).float().div(255)
            pixels = ((pixels - self._mean) / self._std).to(self.dtype)
            tokens = self.model(pixel_values=pixels).last_hidden_state[:, self.num_prefix:]
            self.embed_passes += 1
            tokens = torch.nn.functional.normalize(tokens.float(), dim=-1)
            for j, i in enumerate(indices):
                features[i] = tokens[j].view(h // self.patch_size, w // self.patch_size, -1)
        return features

    def _embed_images(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """Return each image's normalised patch-feature grid and original size."""
        features = self._patch_features([self._resize(image, self.image_size) for image in images])
        return [{"features": f.half().cpu(), "size": image.size} for f, image in zip(features, images)]

    def _embed_probes(self, probes: list[Image.Image]) -> list[torch.Tensor]:
        """Return each probe's prototype: the normalised mean patch feature inside the trimmed border."""
        prototypes = []
        for grid in self._patch_features([self._resize(probe, self.probe_size) for probe in probes]):
            rows, cols = grid.shape[:2]
            dr, dc = int(rows * self.probe_border), int(cols * self.probe_border)
            core = grid[dr:rows - dr or None, dc:cols - dc or None]
            prototypes.append(torch.nn.functional.normalize(core.reshape(-1, grid.shape[-1]).mean(0), dim=0).half().cpu())
        return prototypes

    def _score(self, image_embedding: dict[str, Any], probe_embeddings: list[torch.Tensor]) -> list[ProbeResult]:
        """Score all probes against one image's patches with one matrix product."""
        grid = image_embedding["features"].to(self.device).float()
        rows, cols, dim = grid.shape
        sims = grid.reshape(-1, dim) @ torch.stack(probe_embeddings).to(self.device).float().T  # [patches, probes]
        presence = sims.topk(min(self.top_k, len(sims)), dim=0).values.mean(0)
        calibrate = lambda x: ((x - self.sim_offset) / self.sim_temperature).sigmoid()
        maps = calibrate(sims).T.reshape(-1, rows, cols).cpu()
        width, height = image_embedding["size"]
        cell = (width / cols, height / rows)
        return [
            ProbeResult(probability=calibrate(presence[j]).item(), boxes=self._regions(maps[j], cell))
            for j in range(len(probe_embeddings))
        ]

    def _regions(self, prob_map: torch.Tensor, cell: tuple[float, float]) -> list[DetectedBox]:
        """Return boxes around 4-connected regions of patches scoring at least `box_threshold`.

        Each box spans its region's patches, in image pixels; its score is
        the region's highest patch score. The best `max_boxes` are kept.
        """
        rows, cols = prob_map.shape
        above = (prob_map >= self.box_threshold).tolist()
        seen = [[False] * cols for _ in range(rows)]
        boxes = []
        for r0 in range(rows):
            for c0 in range(cols):
                if not above[r0][c0] or seen[r0][c0]:
                    continue
                seen[r0][c0] = True
                stack, cells = [(r0, c0)], []
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if 0 <= nr < rows and 0 <= nc < cols and above[nr][nc] and not seen[nr][nc]:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
                rs, cs = [r for r, _ in cells], [c for _, c in cells]
                score = max(prob_map[r, c].item() for r, c in cells)
                boxes.append(DetectedBox(round(min(cs) * cell[0], 1), round(min(rs) * cell[1], 1),
                                         round((max(cs) + 1) * cell[0], 1), round((max(rs) + 1) * cell[1], 1),
                                         score))
        return sorted(boxes, key=lambda b: -b.score)[:self.max_boxes]


if __name__ == "__main__":
    run_cli(ImageProber_DINOv3)
