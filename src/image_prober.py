"""Abstract interface for models that answer "is probe i present in image j?".

`ImageProber` defines one public method, `image_probe(images, probes)`, which
scores every probe against every image. Concrete subclasses supply three
model-specific steps: embedding images, embedding probes, and scoring one
image embedding against a set of probe embeddings. The base class supplies the
shared machinery:

- weight download into the weights folder, skipped when already present;
- an embedding cache (in memory, plus optionally on disk) keyed by image
  content, so each image or probe is embedded at most once;
- batching, so the expensive model runs once per batch of images or probes,
  giving N + M embedding passes rather than N x M;
- box post-processing (thresholding and non-maximum suppression);
- an evaluation CLI (`run_cli`) that scores a prober against the sample
  datasets' hidden ground truth.
"""

import argparse
import hashlib
import time
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Self

import torch
import torchvision
import yaml
from huggingface_hub import snapshot_download
from PIL import Image

# An image or probe: a file path, or an image already in memory.
ImageInput = str | Path | Image.Image
# Result keys: the path (as a string) for file inputs, the list index for in-memory images.
ResultKey = str | int
# Marker file written once a weight download has finished.
DOWNLOAD_MARKER = ".download_complete"


@dataclass(frozen=True)
class DetectedBox:
    """One location where a probe was found, in the input image's pixel coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float
    score: float


@dataclass(frozen=True)
class ProbeResult:
    """How likely a probe is present in an image, and where (if the model localises)."""

    probability: float
    boxes: list[DetectedBox] = field(default_factory=list)


class EmbeddingCache:
    """Two-level embedding cache: an in-memory LRU, backed by optional `.pt` files on disk.

    Values are whatever a prober's embed step returns (typically a dict of
    CPU tensors). Keys should identify both the model and the image content.
    """

    def __init__(self, memory_items: int, disk_dir: Path | None) -> None:
        """Keep up to `memory_items` values in memory; persist to `disk_dir` if given."""
        self.memory_items = memory_items
        self.disk_dir = disk_dir
        self._memory: OrderedDict[str, Any] = OrderedDict()
        if disk_dir is not None:
            disk_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        """Return the cached value for `key`, or None on a miss."""
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        if self.disk_dir is not None:
            path = self.disk_dir / f"{key}.pt"
            if path.exists():
                value = torch.load(path, weights_only=False)
                self._remember(key, value)
                return value
        return None

    def put(self, key: str, value: Any) -> None:
        """Store `value` under `key` in memory and, if enabled, on disk."""
        self._remember(key, value)
        if self.disk_dir is not None:
            torch.save(value, self.disk_dir / f"{key}.pt")

    def _remember(self, key: str, value: Any) -> None:
        """Insert into the in-memory LRU, evicting the oldest entries beyond capacity."""
        self._memory[key] = value
        self._memory.move_to_end(key)
        while len(self._memory) > self.memory_items:
            self._memory.popitem(last=False)


class ImageProber(ABC):
    """Base class for probe-in-image models.

    Subclasses implement `_embed_images`, `_embed_probes`, `_score` and
    `cache_tag`, and set `config_section` to the name of their section in
    `config.yaml`. Embeddings are opaque to this class: it only batches,
    caches and routes them.
    """

    # Name of the subclass's section in config.yaml.
    config_section: ClassVar[str]

    def __init__(
        self,
        weights_dir: str | Path,
        cache_dir: str | Path | None = None,
        device: str = "auto",
        batch_size: int = 8,
        memory_cache_items: int = 256,
    ) -> None:
        """Set up shared state; subclasses load their model after calling this."""
        self.weights_dir = Path(weights_dir)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   "cpu" if device == "auto" else device)
        self.batch_size = batch_size
        self.cache = EmbeddingCache(memory_cache_items, Path(cache_dir) if cache_dir else None)
        # Model passes run by the embed steps and by the (lighter) scoring step.
        # Subclasses increment these at each forward pass; the evaluation CLI reports them.
        self.embed_passes = 0
        self.score_passes = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Build a prober from a parsed `config.yaml`.

        Shared settings come from the `image_prober` section and model-specific
        settings from `cls.config_section`; `general.weights_dir` locates weights.
        """
        return cls(weights_dir=config["general"]["weights_dir"], **config["image_prober"],
                   **config[cls.config_section])

    # ----- public API -------------------------------------------------------------

    def image_probe(
        self, images: Sequence[ImageInput], probes: Sequence[ImageInput]
    ) -> dict[ResultKey, dict[ResultKey, ProbeResult]]:
        """Score every probe against every image.

        Returns `{image_key: {probe_key: ProbeResult}}` with one entry per
        image and, inside it, one per probe. Keys are the input path as a
        string, or the list index for in-memory images. Probes are embedded
        once; images are embedded in batches and scored against all probes
        before the next batch, so memory stays bounded for large image sets.
        """
        probe_keys = [_result_key(p, i) for i, p in enumerate(probes)]
        probe_embeddings = self._embeddings("probe", probes)
        context = self._image_context(probe_embeddings, [_content_hash(p) for p in probes])
        results: dict[ResultKey, dict[ResultKey, ProbeResult]] = {}
        for start in range(0, len(images), self.batch_size):
            chunk = images[start:start + self.batch_size]
            for offset, embedding in enumerate(self._embeddings("image", chunk, context)):
                key = _result_key(chunk[offset], start + offset)
                results[key] = dict(zip(probe_keys, self._score(embedding, probe_embeddings)))
        return results

    # ----- subclass hooks ---------------------------------------------------------

    @property
    @abstractmethod
    def cache_tag(self) -> str:
        """Short, filename-safe string identifying the model and any settings that change embeddings."""

    @abstractmethod
    def _embed_images(self, images: list[Image.Image]) -> list[Any]:
        """Embed a batch of target images, counting passes in `embed_passes`; return one embedding per image."""

    @abstractmethod
    def _embed_probes(self, probes: list[Image.Image]) -> list[Any]:
        """Embed a batch of probe crops, counting passes in `embed_passes`; return one embedding per probe."""

    @abstractmethod
    def _score(self, image_embedding: Any, probe_embeddings: list[Any]) -> list[ProbeResult]:
        """Score one image embedding against every probe embedding, in probe order."""

    def _image_context(self, probe_embeddings: list[Any], probe_hashes: list[str]) -> str:
        """Prepare for embedding images against this probe set; return a string added to image cache keys.

        Called once per `image_probe` call, after the probes are embedded.
        Most probers embed images independently of the probes and return ""
        (the default), so image embeddings are reused across probe sets. A
        prober whose image embeddings depend on the probes (e.g. one that
        draws the probes onto each image) keeps what it needs here and returns
        a key identifying the probe set.
        """
        return ""

    # ----- shared machinery -------------------------------------------------------

    def ensure_weights(
        self, repo_id: str, allow_patterns: list[str] | None = None, ignore_patterns: list[str] | None = None
    ) -> Path:
        """Return the local folder holding `repo_id`'s weights, downloading them only if missing.

        Weights live in `<weights_dir>/<org>--<name>/`. A marker file is written
        after a complete download, so an interrupted download is retried and a
        finished one is never fetched again.
        """
        local = self.weights_dir / repo_id.replace("/", "--")
        if (local / DOWNLOAD_MARKER).exists():
            print(f"[{type(self).__name__}] using weights in {local}")
            return local
        print(f"[{type(self).__name__}] downloading {repo_id} to {local}")
        snapshot_download(repo_id, local_dir=local, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
        (local / DOWNLOAD_MARKER).touch()
        return local

    def _embeddings(self, kind: str, items: Sequence[ImageInput], context: str = "") -> list[Any]:
        """Return embeddings for `items` (`kind` is "image" or "probe"), embedding only cache misses.

        `context` (from `_image_context`) is added to the cache keys. Misses
        are embedded in batches of `batch_size`, each batch being one call to
        the subclass's embed step (normally a single model pass).
        """
        suffix = f"-{context}" if context else ""
        keys = [f"{self.cache_tag}-{kind}-{_content_hash(item)}{suffix}" for item in items]
        found = [self.cache.get(k) for k in keys]
        missing = [i for i, value in enumerate(found) if value is None]
        embed = self._embed_images if kind == "image" else self._embed_probes
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            with torch.inference_mode():
                values = embed([_load_image(items[i]) for i in batch])
            for i, value in zip(batch, values):
                self.cache.put(keys[i], value)
                found[i] = value
        return found


def select_boxes(
    boxes_xyxy: torch.Tensor, scores: torch.Tensor, threshold: float, nms_iou: float, max_boxes: int
) -> list[DetectedBox]:
    """Keep boxes scoring at least `threshold`, suppress overlaps above `nms_iou`, return the best `max_boxes`.

    Coordinates are rounded to 0.1 px; scores keep full precision, since
    near-saturated calibrated scores would otherwise tie and lose their ranking.
    """
    keep = scores >= threshold
    boxes_xyxy, scores = boxes_xyxy[keep].float(), scores[keep].float()
    if not len(scores):
        return []
    order = torchvision.ops.nms(boxes_xyxy, scores, nms_iou)[:max_boxes]
    return [DetectedBox(*(round(v, 1) for v in boxes_xyxy[i].tolist()), score=scores[i].item())
            for i in order.tolist()]


def _result_key(item: ImageInput, index: int) -> ResultKey:
    """Return the result key for an input: its path as a string, or its index if in memory."""
    return index if isinstance(item, Image.Image) else str(item)


def _load_image(item: ImageInput) -> Image.Image:
    """Return `item` as an RGB PIL image."""
    image = item if isinstance(item, Image.Image) else Image.open(item)
    return image.convert("RGB")


def _content_hash(item: ImageInput) -> str:
    """Hash an input's content (file bytes, or pixels for in-memory images) for cache keys."""
    if isinstance(item, Image.Image):
        data = f"{item.mode}{item.size}".encode() + item.tobytes()
    else:
        data = Path(item).read_bytes()
    return hashlib.sha1(data).hexdigest()[:20]


# ----- evaluation against the sample datasets -------------------------------------


def _average_precision(scores: list[float], positives: list[bool]) -> float:
    """Average precision of ranking by `scores` (mean precision at each positive's rank)."""
    ranked = sorted(zip(scores, positives), key=lambda sp: -sp[0])
    hits, total = 0, 0.0
    for rank, (_, positive) in enumerate(ranked, start=1):
        if positive:
            hits += 1
            total += hits / rank
    return total / hits if hits else float("nan")


def _roc_auc(scores: list[float], positives: list[bool]) -> float:
    """Probability that a random positive outscores a random negative (ties count half)."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _best_f1(scores: list[float], positives: list[bool]) -> tuple[float, float]:
    """Return (best F1, threshold achieving it) over all score thresholds."""
    best = (0.0, 1.0)
    n_pos = sum(positives)
    for threshold in sorted(set(scores)):
        predicted = [s >= threshold for s in scores]
        tp = sum(p and g for p, g in zip(predicted, positives))
        if tp:
            precision, recall = tp / sum(predicted), tp / n_pos
            best = max(best, (2 * precision * recall / (precision + recall), threshold))
    return best


def _iou(a: DetectedBox, xywh: list[float]) -> float:
    """IoU between a detected box and a ground-truth box given as [x, y, w, h]."""
    bx0, by0, bx1, by1 = xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]
    iw = max(0.0, min(a.x1, bx1) - max(a.x0, bx0))
    ih = max(0.0, min(a.y1, by1) - max(a.y0, by0))
    inter = iw * ih
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + xywh[2] * xywh[3] - inter
    return inter / union if union > 0 else 0.0


def evaluate_dataset(prober: ImageProber, dataset_dir: Path, box_iou: float) -> dict[str, Any]:
    """Probe every image of a sample dataset with all its probes and score against `images.yaml`.

    A class's score for an image is the highest probability among that
    class's probes. Image-level metrics per class: average precision, ROC
    AUC and best F1. Box recall is the share of ground-truth boxes matched
    (IoU >= `box_iou`) by any box returned for a probe of the same class.
    """
    probes_meta = yaml.safe_load((dataset_dir / "probes.yaml").read_text())
    images_meta = yaml.safe_load((dataset_dir / "images.yaml").read_text())
    probe_paths = [dataset_dir / p["file"] for p in probes_meta["probes"]]
    image_paths = [dataset_dir / r["file"] for r in images_meta["images"]]

    passes_before = prober.embed_passes, prober.score_passes
    start = time.perf_counter()
    results = prober.image_probe(image_paths, probe_paths)
    seconds = time.perf_counter() - start

    probe_labels = {str(path): p["label"] for path, p in zip(probe_paths, probes_meta["probes"])}
    per_class: dict[str, Any] = {}
    for label in images_meta["labels"]:
        scores, positives, matched, total_boxes = [], [], 0, 0
        for path, record in zip(image_paths, images_meta["images"]):
            hits = [r for key, r in results[str(path)].items() if probe_labels[key] == label]
            scores.append(max(r.probability for r in hits))
            positives.append(label in record["labels"])
            found = [b for r in hits for b in r.boxes]
            for gt in (b for b in record["boxes"] if b["label"] == label):
                total_boxes += 1
                matched += any(_iou(b, gt["bbox_xywh"]) >= box_iou for b in found)
        f1, threshold = _best_f1(scores, positives)
        per_class[label] = {
            "average_precision": round(_average_precision(scores, positives), 3),
            "roc_auc": round(_roc_auc(scores, positives), 3),
            "best_f1": round(f1, 3),
            "best_f1_threshold": round(threshold, 4),
            "box_recall": round(matched / total_boxes, 3) if total_boxes else None,
        }
    mean = {m: round(sum(c[m] for c in per_class.values()) / len(per_class), 3)
            for m in ("average_precision", "roc_auc", "best_f1")}
    return {
        "dataset": dataset_dir.name,
        "images": len(image_paths),
        "probes": len(probe_paths),
        "embed_passes": prober.embed_passes - passes_before[0],
        "score_passes": prober.score_passes - passes_before[1],
        "seconds": round(seconds, 1),
        "mean": mean,
        "per_class": per_class,
        "results": {
            Path(img).name: {Path(p).name: {"probability": round(r.probability, 4),
                                            "boxes": [[b.x0, b.y0, b.x1, b.y1, round(b.score, 6)] for b in r.boxes]}
                             for p, r in per_probe.items()}
            for img, per_probe in results.items()
        },
    }


def run_cli(prober_cls: type[ImageProber]) -> None:
    """Command-line entry point shared by the concrete probers' `__main__` blocks.

    Builds `prober_cls` from `config.yaml`, evaluates it on the sample
    datasets, prints per-class metrics and writes the full results to
    `<output_dir>/<config_section>_<dataset>.yaml`.
    """
    parser = argparse.ArgumentParser(description=f"Evaluate {prober_cls.__name__} on the sample datasets.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    parser.add_argument("--datasets", nargs="+", default=None, help="dataset names (default: all)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    datasets_dir = Path(config["sample_data"]["datasets_dir"])
    out_dir = Path(config["general"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    box_iou = config["image_prober_eval"]["box_iou"]

    prober = prober_cls.from_config(config)
    folders = sorted(p for p in datasets_dir.iterdir() if (p / "images.yaml").exists())
    for folder in folders:
        if args.datasets and folder.name not in args.datasets:
            continue
        report = evaluate_dataset(prober, folder, box_iou)
        out = out_dir / f"{prober_cls.config_section}_{folder.name}.yaml"
        out.write_text(yaml.safe_dump(report, sort_keys=False, default_flow_style=None, width=140))
        print(f"[{folder.name}] {report['images']} images x {report['probes']} probes: "
              f"{report['embed_passes']} embed + {report['score_passes']} score passes, {report['seconds']} s -> {out}")
        print(f"    {'class':<20} {'AP':>6} {'AUC':>6} {'F1':>6} {'thresh':>7} {'box rec':>8}")
        for label, m in report["per_class"].items():
            box_recall = "-" if m["box_recall"] is None else f"{m['box_recall']:.3f}"
            print(f"    {label:<20} {m['average_precision']:>6.3f} {m['roc_auc']:>6.3f} "
                  f"{m['best_f1']:>6.3f} {m['best_f1_threshold']:>7.4f} {box_recall:>8}")
        m = report["mean"]
        print(f"    {'mean':<20} {m['average_precision']:>6.3f} {m['roc_auc']:>6.3f} {m['best_f1']:>6.3f}")
