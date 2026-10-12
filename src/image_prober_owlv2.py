"""OWLv2 image-guided detection as an `ImageProber`.

OWLv2 (Minderer et al., 2023) predicts, for every patch of an image, a box
and a class embedding; a query embedding is scored against every patch with
a dot product plus a learned per-patch shift and scale. Because the boxes
and patch embeddings do not depend on the query, each image and each probe
needs exactly one pass through the vision transformer, and scoring all
probes against an image is a single small matrix product.

A probe's query embedding is the class embedding of the patch whose
predicted box best covers the probe: among boxes with IoU of at least
`query_iou_fraction` x the best IoU with the probe's extent, the one with the
highest objectness. (Recent `transformers` uses objectness alone, which can
pick a part of the object or a background item.)

Image-derived queries give much larger logits than the text queries OWLv2's
sigmoid was trained for, so raw scores saturate near 1 even for absent
probes. Patch logits are therefore calibrated as
`sigmoid((logit - logit_offset) / logit_temperature)` before use. This
changes probabilities and box thresholds but not the ranking of images.
"""

from pathlib import Path
from typing import Any

import torch
import torchvision
from PIL import Image
from transformers import Owlv2ForObjectDetection
from transformers import Owlv2ImageProcessor

from .image_prober import ImageProber
from .image_prober import ProbeResult
from .image_prober import run_cli
from .image_prober import select_boxes

# Files needed to load the model; the duplicate pytorch_model.bin and tokenizer are skipped.
WEIGHT_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]


class ImageProber_OwlV2(ImageProber):
    """Probe images with OWLv2 image-conditioned detection.

    Image embeddings hold, per patch, the normalised class embedding, the
    logit shift and scale, and the predicted box in the image's pixels.
    Probe embeddings are a single normalised query vector. A probe's
    probability for an image is the highest calibrated patch score; its
    boxes are the patches scoring at least `box_threshold`, after
    non-maximum suppression.
    """

    config_section = "image_prober_owlv2"

    def __init__(
        self,
        repo_id: str = "google/owlv2-base-patch16-ensemble",
        dtype: str = "float16",
        box_threshold: float = 0.5,
        nms_iou: float = 0.3,
        max_boxes: int = 20,
        query_iou_fraction: float = 0.8,
        logit_offset: float = 0.0,
        logit_temperature: float = 1.0,
        **common: Any,
    ) -> None:
        """Load OWLv2 from the weights folder (downloading once); `common` goes to `ImageProber`."""
        super().__init__(**common)
        self.repo_id = repo_id
        self.box_threshold = box_threshold
        self.nms_iou = nms_iou
        self.max_boxes = max_boxes
        self.query_iou_fraction = query_iou_fraction
        self.logit_offset = logit_offset
        self.logit_temperature = logit_temperature
        path = self.ensure_weights(repo_id, allow_patterns=WEIGHT_FILES)
        self.processor = Owlv2ImageProcessor.from_pretrained(path)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.model = Owlv2ForObjectDetection.from_pretrained(path, dtype=self.dtype).to(self.device).eval()

    @property
    def cache_tag(self) -> str:
        """Model name plus the query-selection setting (probe embeddings depend on it)."""
        return f"owlv2-{Path(self.repo_id).name}-q{self.query_iou_fraction}"

    def _patch_outputs(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """Run the vision transformer once on `images` and return per-patch head outputs.

        Boxes are xyxy, normalised to the padded square the processor builds
        (images are padded at the bottom/right to square, then resized).
        """
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        feature_map = self.model.image_embedder(pixel_values=pixels.to(self.device, self.dtype))[0]
        self.embed_passes += 1
        batch, height, width, dim = feature_map.shape
        feats = feature_map.reshape(batch, height * width, dim)
        head = self.model.class_head
        class_embeds = head.dense0(feats)
        return {
            "class_embeds": class_embeds / (torch.linalg.norm(class_embeds, dim=-1, keepdim=True) + 1e-6),
            "logit_shift": head.logit_shift(feats)[..., 0],
            "logit_scale": (head.elu(head.logit_scale(feats)) + 1)[..., 0],
            "boxes": torchvision.ops.box_convert(self.model.box_predictor(feats, feature_map), "cxcywh", "xyxy"),
            "objectness": self.model.objectness_predictor(feats),
        }

    def _embed_images(self, images: list[Image.Image]) -> list[dict[str, torch.Tensor]]:
        """Return per-patch class embeddings, logit shift/scale and pixel-space boxes for each image."""
        out = self._patch_outputs(images)
        embeddings = []
        for i, image in enumerate(images):
            side = max(image.size)  # boxes are relative to the padded square
            embeddings.append({
                "class_embeds": out["class_embeds"][i].half().cpu(),
                "logit_shift": out["logit_shift"][i].float().cpu(),
                "logit_scale": out["logit_scale"][i].float().cpu(),
                "boxes": (out["boxes"][i].float() * side).cpu(),
            })
        return embeddings

    def _embed_probes(self, probes: list[Image.Image]) -> list[torch.Tensor]:
        """Return one normalised query embedding per probe (see the module docstring for the choice)."""
        out = self._patch_outputs(probes)
        queries = []
        for i, probe in enumerate(probes):
            side = max(probe.size)
            extent = torch.tensor([[0.0, 0.0, probe.width / side, probe.height / side]], device=self.device)
            ious = torchvision.ops.box_iou(extent, out["boxes"][i].float())[0]
            candidates = (ious >= ious.max() * self.query_iou_fraction).nonzero()[:, 0]
            best = candidates[out["objectness"][i][candidates].argmax()]
            queries.append(out["class_embeds"][i][best].half().cpu())
        return queries

    def _score(self, image_embedding: dict[str, torch.Tensor], probe_embeddings: list[torch.Tensor]) -> list[ProbeResult]:
        """Score all probes against one image with a single matrix product over patches."""
        emb = {k: v.to(self.device) for k, v in image_embedding.items()}
        queries = torch.stack(probe_embeddings).to(self.device)
        logits = (emb["class_embeds"].float() @ queries.float().T + emb["logit_shift"][:, None]) * emb["logit_scale"][:, None]
        probs = ((logits - self.logit_offset) / self.logit_temperature).sigmoid()  # [patches, probes]
        return [
            ProbeResult(
                probability=probs[:, j].max().item(),
                boxes=select_boxes(emb["boxes"], probs[:, j], self.box_threshold, self.nms_iou, self.max_boxes),
            )
            for j in range(len(probe_embeddings))
        ]


if __name__ == "__main__":
    run_cli(ImageProber_OwlV2)
