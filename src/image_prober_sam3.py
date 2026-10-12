"""SAM 3 promptable concept detection as an `ImageProber`, with probe crops as exemplars.

SAM 3 (Meta, 2025) accepts an exemplar only as a box drawn on the image being
searched: its geometry encoder pools that box's features from the same image.
It is trained only on such same-image exemplars, so its presence head has
learnt that "an exemplar was given" implies "the concept is present". Two
ways of feeding it a probe are implemented (`exemplar_mode`):

- `canvas` (default): each target image is drawn on a 1008 x 1008 canvas
  above a strip of probe tiles, and each probe's tile is boxed as a genuine
  same-image exemplar. Detections centred in the strip are discarded. The
  vision encoder runs once per canvas, i.e. once per image for up to
  `strip_columns` x (rows that fit in `max_strip_fraction`) probes (more
  probes spread over extra canvases). Each (image, probe) pair then needs
  one pass of the DETR encoder/decoder, batched `probe_batch_size` at a
  time. Presence is always ~1 here (the tile itself is present), so a
  probe's probability is its best query score outside the strip.
- `transfer`: each probe is encoded in its own frame (vision encoder +
  geometry encoder on a box covering the crop), and those prompt tokens are
  reused for every image. Cheaper and probe-set independent, but measured
  near chance: the presence head reports ~1 for every image. Kept for
  reference.

The mask decoder is never run; only boxes and scores are produced.
"""

import hashlib
import math
from typing import Any

import torch
import torchvision
from PIL import Image
from transformers import Sam3Model
from transformers import Sam3Processor
from transformers.models.sam3.modeling_sam3 import inverse_sigmoid

from .image_prober import ImageProber
from .image_prober import ProbeResult
from .image_prober import run_cli
from .image_prober import select_boxes

# Text prompt SAM 3's processor substitutes when only boxes are given.
VISUAL_PROMPT = "visual"
# A box covering the whole probe crop, in normalised (cx, cy, w, h); used by transfer mode.
WHOLE_CROP_BOX = [0.5, 0.5, 1.0, 1.0]
# SAM 3's input resolution; canvases are built at this size so the processor does not rescale them.
CANVAS_SIDE = 1008
# Canvas background behind the image and probe tiles.
CANVAS_FILL = (0, 0, 0)
# Resampling for canvas drawing; measured on coco_objects, bicubic gave mean AP 0.80 vs 0.68 for bilinear.
CANVAS_RESAMPLE = Image.Resampling.BICUBIC


class ImageProber_SAM3(ImageProber):
    """Probe images with SAM 3, using probe crops as exemplars (see the module docstring for the modes)."""

    config_section = "image_prober_sam3"

    def __init__(
        self,
        repo_id: str = "facebook/sam3",
        dtype: str = "bfloat16",
        exemplar_mode: str = "canvas",
        strip_columns: int = 10,
        max_strip_fraction: float = 0.25,
        probe_batch_size: int = 20,
        probability: str = "max_score",
        box_threshold: float = 0.5,
        nms_iou: float = 0.5,
        max_boxes: int = 20,
        **common: Any,
    ) -> None:
        """Load SAM 3 from the weights folder (downloading once); `common` goes to `ImageProber`.

        The checkpoint is gated on Hugging Face: accept its licence on the
        model page and log in (`hf auth login`) before the first download.
        `probability` applies to transfer mode only: `max_score` (best query x
        presence) or `presence` (image-level presence alone).
        """
        super().__init__(**common)
        if exemplar_mode not in ("canvas", "transfer"):
            raise ValueError(f"exemplar_mode must be 'canvas' or 'transfer', not {exemplar_mode!r}")
        if probability not in ("max_score", "presence"):
            raise ValueError(f"probability must be 'max_score' or 'presence', not {probability!r}")
        self.repo_id = repo_id
        self.exemplar_mode = exemplar_mode
        self.strip_columns = strip_columns
        self.max_strip_fraction = max_strip_fraction
        self.probe_batch_size = probe_batch_size
        self.probability = probability
        self.box_threshold = box_threshold
        self.nms_iou = nms_iou
        self.max_boxes = max_boxes
        # Canvas layout: square tiles in a strip along the bottom of the canvas.
        self.tile = CANVAS_SIDE // strip_columns
        self.probes_per_canvas = strip_columns * max(1, int(CANVAS_SIDE * max_strip_fraction) // self.tile)
        # Probe tiles for the current probe set, set by `_image_context` in canvas mode.
        self._canvas_probes: list[Image.Image] = []
        # The repo also ships the original (non-transformers) checkpoint; skip it.
        path = self.ensure_weights(repo_id, ignore_patterns=["*.pt", "*.pth", "*.bin"])
        self.processor = Sam3Processor.from_pretrained(path)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.model = Sam3Model.from_pretrained(path, dtype=self.dtype).to(self.device).eval()
        self._prepare_constants()

    def _prepare_constants(self) -> None:
        """Compute the inputs shared by every scoring pass: the text prompt and position encoding.

        The FPN position encoding depends only on the input size, so one
        pass on a blank image provides it; image embeddings then need not
        store it.
        """
        with torch.inference_mode():
            text = self.processor(text=VISUAL_PROMPT, return_tensors="pt").to(self.device)
            self._text_features = self.model.get_text_features(**text).pooler_output
            self._text_mask = text["attention_mask"].bool()
            blank = self.processor(images=Image.new("RGB", (64, 64)), return_tensors="pt")["pixel_values"]
            vision = self.model.get_vision_features(pixel_values=blank.to(self.device, self.dtype))
            self._pos_embed = vision.fpn_position_encoding[-2]

    @property
    def cache_tag(self) -> str:
        """Model name, exemplar mode and canvas layout (which change the embeddings)."""
        name = self.repo_id.split("/")[-1]
        if self.exemplar_mode == "transfer":
            return f"sam3-{name}-transfer"
        return f"sam3-{name}-canvas{self.strip_columns}x{self.probes_per_canvas}-bicubic"

    # ----- shared SAM 3 steps -----------------------------------------------------

    def _vision(self, images: list[Image.Image]) -> Any:
        """Run SAM 3's vision encoder once on a batch of images."""
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        self.embed_passes += 1
        return self.model.get_vision_features(pixel_values=pixels.to(self.device, self.dtype))

    def _exemplar_tokens(self, level: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Encode one exemplar box per batch entry against its feature map.

        `level` is the [B, 256, 72, 72] FPN level the geometry encoder reads;
        `boxes` are [B, 4] normalised (cx, cy, w, h). Returns [B, 2, 256]
        prompt tokens (box token plus the encoder's CLS token).
        """
        batch = len(boxes)
        geometry = self.model.geometry_encoder(
            box_embeddings=boxes[:, None].to(self.device, self.dtype),
            box_mask=torch.ones(batch, 1, dtype=torch.bool, device=self.device),
            box_labels=torch.ones(batch, 1, dtype=torch.long, device=self.device),
            img_feats=(level,),
            img_pos_embeds=(self._pos_embed.expand(batch, -1, -1, -1),),
        )
        return geometry.last_hidden_state

    def _detect(
        self, features: torch.Tensor, probe_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect each probe's concept in one feature map; mirrors `Sam3Model.forward` minus the mask decoder.

        Returns per-query probabilities [B, Q] (query sigmoid, without
        presence), presence probabilities [B] and normalised xyxy boxes [B, Q, 4].
        """
        batch = len(probe_tokens)
        vision = features.to(self.device, self.dtype)[None].expand(batch, -1, -1, -1)
        pos = self._pos_embed.expand(batch, -1, -1, -1)
        prompt = torch.cat([self._text_features.expand(batch, -1, -1), probe_tokens.to(self.device, self.dtype)], dim=1)
        prompt_mask = torch.cat([
            self._text_mask.expand(batch, -1),
            torch.ones(batch, probe_tokens.shape[1], dtype=torch.bool, device=self.device),
        ], dim=1)
        encoded = self.model.detr_encoder(
            vision_features=[vision], text_features=prompt, vision_pos_embeds=[pos], text_mask=prompt_mask
        )
        decoded = self.model.detr_decoder(
            vision_features=encoded.last_hidden_state,
            text_features=encoded.text_features,
            vision_pos_encoding=encoded.pos_embeds_flattened,
            text_mask=prompt_mask,
            spatial_shapes=encoded.spatial_shapes,
        )
        self.score_passes += 1
        # Only the last decoder layer's outputs are needed.
        hidden = decoded.intermediate_hidden_states[-1:]
        offsets = self.model.detr_decoder.box_head(hidden)
        boxes = (inverse_sigmoid(decoded.reference_boxes[-1:]) + offsets).sigmoid()[0]
        logits = self.model.dot_product_scoring(
            decoder_hidden_states=hidden, text_features=encoded.text_features, text_mask=prompt_mask
        )[0, ..., 0]
        presence = decoded.presence_logits[-1].float().sigmoid().reshape(batch)
        return logits.float().sigmoid(), presence, torchvision.ops.box_convert(boxes.float(), "cxcywh", "xyxy")

    # ----- ImageProber hooks ------------------------------------------------------

    def _embed_probes(self, probes: list[Image.Image]) -> list[Any]:
        """Canvas mode: the probe image itself (drawn onto canvases later). Transfer mode: prompt tokens."""
        if self.exemplar_mode == "canvas":
            return [probe.copy() for probe in probes]
        vision = self._vision(probes)
        boxes = torch.tensor([WHOLE_CROP_BOX] * len(probes))
        return [t.cpu() for t in self._exemplar_tokens(vision.fpn_hidden_states[-2], boxes)]

    def _image_context(self, probe_embeddings: list[Any], probe_hashes: list[str]) -> str:
        """Canvas mode: keep the probe tiles and key image embeddings by the probe set."""
        if self.exemplar_mode == "transfer":
            return ""
        self._canvas_probes = probe_embeddings
        return "probes" + hashlib.sha1("".join(probe_hashes).encode()).hexdigest()[:16]

    def _embed_images(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """Transfer mode: each image's detector feature map. Canvas mode: per probe group, the
        canvas feature map plus each probe tile's exemplar tokens.
        """
        if self.exemplar_mode == "transfer":
            features = self._vision(images).fpn_hidden_states[-2]
            return [{"features": features[i].cpu(), "size": image.size} for i, image in enumerate(images)]

        groups = [self._canvas_probes[i:i + self.probes_per_canvas]
                  for i in range(0, len(self._canvas_probes), self.probes_per_canvas)]
        layouts = [[self._canvas(image, group) for group in groups] for image in images]
        flat = [layout for per_image in layouts for layout in per_image]
        level = torch.cat([self._vision([c for c, _, _ in flat[i:i + self.batch_size]]).fpn_hidden_states[-2]
                           for i in range(0, len(flat), self.batch_size)])
        embeddings, k = [], 0
        for image, per_image in zip(images, layouts):
            entry: dict[str, Any] = {"size": image.size, "groups": []}
            for _, scale, tile_boxes in per_image:
                tokens = self._exemplar_tokens(level[k][None].expand(len(tile_boxes), -1, -1, -1), tile_boxes)
                entry["groups"].append({"features": level[k].cpu(), "tokens": tokens.cpu()})
                entry["scale"] = scale
                k += 1
            embeddings.append(entry)
        return embeddings

    def _canvas(self, image: Image.Image, probes: list[Image.Image]) -> tuple[Image.Image, float, torch.Tensor]:
        """Draw `image` above a strip of `probes` tiles on a CANVAS_SIDE square.

        Returns the canvas, the image's resize factor, and each tile's box as
        normalised (cx, cy, w, h). The image is fitted into the area above the
        strip, top-left aligned.
        """
        rows = math.ceil(len(probes) / self.strip_columns)
        strip_top = CANVAS_SIDE - rows * self.tile
        canvas = Image.new("RGB", (CANVAS_SIDE, CANVAS_SIDE), CANVAS_FILL)
        scale = min(CANVAS_SIDE / image.width, strip_top / image.height)
        canvas.paste(image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                  CANVAS_RESAMPLE), (0, 0))
        boxes = []
        for j, probe in enumerate(probes):
            fit = min(self.tile / probe.width, self.tile / probe.height)
            w, h = max(1, round(probe.width * fit)), max(1, round(probe.height * fit))
            x0, y0 = (j % self.strip_columns) * self.tile, strip_top + (j // self.strip_columns) * self.tile
            canvas.paste(probe.resize((w, h), CANVAS_RESAMPLE), (x0, y0))
            boxes.append([(x0 + w / 2) / CANVAS_SIDE, (y0 + h / 2) / CANVAS_SIDE, w / CANVAS_SIDE, h / CANVAS_SIDE])
        return canvas, scale, torch.tensor(boxes)

    def _score(self, image_embedding: dict[str, Any], probe_embeddings: list[Any]) -> list[ProbeResult]:
        """Run the DETR encoder/decoder for one image against every probe, `probe_batch_size` at a time."""
        width, height = image_embedding["size"]
        results = []
        with torch.inference_mode():
            if self.exemplar_mode == "transfer":
                scale = torch.tensor([width, height, width, height], device=self.device)
                for start in range(0, len(probe_embeddings), self.probe_batch_size):
                    tokens = torch.stack(probe_embeddings[start:start + self.probe_batch_size])
                    queries, presence, boxes = self._detect(image_embedding["features"], tokens)
                    scores = queries * presence[:, None]
                    for j in range(len(tokens)):
                        probability = presence[j] if self.probability == "presence" else scores[j].max()
                        results.append(self._result(probability.item(), boxes[j] * scale, scores[j]))
                return results

            # Canvas mode: canvas-normalised boxes -> canvas pixels -> image pixels.
            to_image = CANVAS_SIDE / image_embedding["scale"]
            limit = torch.tensor([width, height, width, height], device=self.device, dtype=torch.float32)
            for group in image_embedding["groups"]:
                for start in range(0, len(group["tokens"]), self.probe_batch_size):
                    queries, _, boxes = self._detect(group["features"], group["tokens"][start:start + self.probe_batch_size])
                    # Keep queries centred on the image area, not on the probe strip or the padding beside it.
                    centre_x, centre_y = (boxes[..., 0] + boxes[..., 2]) / 2 * to_image, (boxes[..., 1] + boxes[..., 3]) / 2 * to_image
                    queries = queries * ((centre_x < width) & (centre_y < height))
                    pixel_boxes = torch.minimum((boxes * to_image).clamp(min=0), limit)
                    for j in range(len(queries)):
                        results.append(self._result(queries[j].max().item(), pixel_boxes[j], queries[j]))
        return results

    def _result(self, probability: float, boxes_xyxy: torch.Tensor, scores: torch.Tensor) -> ProbeResult:
        """Build a ProbeResult from per-query boxes (image pixels) and scores."""
        return ProbeResult(
            probability=probability,
            boxes=select_boxes(boxes_xyxy, scores, self.box_threshold, self.nms_iou, self.max_boxes),
        )


if __name__ == "__main__":
    run_cli(ImageProber_SAM3)
