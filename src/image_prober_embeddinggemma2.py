"""EmbeddingGemma 2 tile retrieval as an `ImageProber`.

EmbeddingGemma 2 (Google DeepMind, 2026; Apache-2.0) maps text, images,
audio and video into one 768-d space, but gives a single vector per input,
with no per-patch features or boxes. A small object barely moves a whole
image's vector, so each image is cut into overlapping square tiles at
several scales (`tile_scales`, as fractions of the image's shorter side,
plus the whole image), and every tile is embedded:

- image embedding: the normalised embedding of every tile, plus the tile
  boxes; tiles are encoded in batches of `tile_batch_size`;
- probe embedding: the normalised embedding of the probe crop;
- score: cosine similarity of the probe with every tile. The presence score
  is the best tile similarity, calibrated as
  `sigmoid((similarity - sim_offset) / sim_temperature)`. Boxes are the
  tiles scoring at least `box_threshold`, after NMS, so they are coarse.

Image embeddings do not depend on the probes, so they are cached and reused.
The cost per image is the number of tiles (about 20 at the default scales)
rather than one pass.
"""

import math
from typing import Any

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from .image_prober import ImageProber
from .image_prober import ProbeResult
from .image_prober import run_cli
from .image_prober import select_boxes


class ImageProber_EmbeddingGemma2(ImageProber):
    """Probe images by matching the probe's embedding against multi-scale image tiles."""

    config_section = "image_prober_embeddinggemma2"

    def __init__(
        self,
        repo_id: str = "google/embeddinggemma-2",
        dtype: str = "bfloat16",
        embedding_dim: int = 768,
        tile_scales: list[float] | None = None,
        tile_overlap: float = 0.5,
        tile_batch_size: int = 32,
        sim_offset: float = 0.72,
        sim_temperature: float = 0.02,
        box_threshold: float = 0.5,
        nms_iou: float = 0.5,
        max_boxes: int = 20,
        **common: Any,
    ) -> None:
        """Load EmbeddingGemma 2 from the weights folder (downloading once); `common` goes to `ImageProber`.

        `embedding_dim` may be 768, 512, 256 or 128 (Matryoshka truncation).
        The model card warns float16 gives NaN or degraded embeddings, so use
        bfloat16 (or float32).
        """
        super().__init__(**common)
        if dtype == "float16":
            raise ValueError("EmbeddingGemma 2 does not support float16; use bfloat16 or float32")
        self.repo_id = repo_id
        self.embedding_dim = embedding_dim
        self.tile_scales = tile_scales if tile_scales is not None else [0.5, 0.33]
        self.tile_overlap = tile_overlap
        self.tile_batch_size = tile_batch_size
        self.sim_offset = sim_offset
        self.sim_temperature = sim_temperature
        self.box_threshold = box_threshold
        self.nms_iou = nms_iou
        self.max_boxes = max_boxes
        path = self.ensure_weights(repo_id)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.model = SentenceTransformer(str(path), device=str(self.device), truncate_dim=embedding_dim,
                                         model_kwargs={"dtype": self.dtype})

    @property
    def cache_tag(self) -> str:
        """Model name plus the tiling and embedding size, which change the embeddings."""
        scales = "_".join(f"{s:g}" for s in self.tile_scales)
        return f"egemma2-{self.repo_id.split('/')[-1]}-d{self.embedding_dim}-t{scales}-o{self.tile_overlap:g}"

    def _encode(self, images: list[Image.Image]) -> torch.Tensor:
        """Embed images in batches of `tile_batch_size`; return [N, dim] normalised float vectors on CPU."""
        chunks = []
        for start in range(0, len(images), self.tile_batch_size):
            batch = [{"image": image} for image in images[start:start + self.tile_batch_size]]
            vectors = self.model.encode(batch, batch_size=len(batch), convert_to_tensor=True, normalize_embeddings=True,
                                        show_progress_bar=False)
            self.embed_passes += 1
            chunks.append(vectors.float().cpu())
        return torch.cat(chunks)

    def _tiles(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        """Return the whole image plus overlapping square tiles at each scale, as pixel (x0, y0, x1, y1)."""
        width, height = image.size
        boxes = [(0, 0, width, height)]
        for scale in self.tile_scales:
            side = max(1, round(min(width, height) * scale))
            step = max(1, round(side * (1 - self.tile_overlap)))
            cols = max(1, math.ceil((width - side) / step) + 1)
            rows = max(1, math.ceil((height - side) / step) + 1)
            for r in range(rows):
                for c in range(cols):
                    # Clamp the last row/column so tiles stay inside the image.
                    x0, y0 = min(c * step, width - side), min(r * step, height - side)
                    boxes.append((x0, y0, x0 + side, y0 + side))
        return list(dict.fromkeys(boxes))

    def _embed_images(self, images: list[Image.Image]) -> list[dict[str, torch.Tensor]]:
        """Return every tile's embedding and box for each image (all tiles of the batch encoded together)."""
        tiles = [self._tiles(image) for image in images]
        crops = [image.crop(box) for image, boxes in zip(images, tiles) for box in boxes]
        vectors = self._encode(crops)
        embeddings, k = [], 0
        for boxes in tiles:
            embeddings.append({"vectors": vectors[k:k + len(boxes)].half(),
                               "boxes": torch.tensor(boxes, dtype=torch.float32)})
            k += len(boxes)
        return embeddings

    def _embed_probes(self, probes: list[Image.Image]) -> list[torch.Tensor]:
        """Return one normalised embedding per probe crop."""
        return list(self._encode(probes).half())

    def _score(self, image_embedding: dict[str, torch.Tensor], probe_embeddings: list[torch.Tensor]) -> list[ProbeResult]:
        """Score all probes against one image's tiles with one matrix product."""
        sims = image_embedding["vectors"].float() @ torch.stack(probe_embeddings).float().T  # [tiles, probes]
        probs = ((sims - self.sim_offset) / self.sim_temperature).sigmoid()
        return [
            ProbeResult(
                probability=probs[:, j].max().item(),
                boxes=select_boxes(image_embedding["boxes"], probs[:, j], self.box_threshold, self.nms_iou,
                                   self.max_boxes),
            )
            for j in range(len(probe_embeddings))
        ]


if __name__ == "__main__":
    run_cli(ImageProber_EmbeddingGemma2)
