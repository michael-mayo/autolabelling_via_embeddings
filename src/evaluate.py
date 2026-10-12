"""Box-level evaluation of any `ImageProber`, using only its public interface.

`Evaluate` runs a prober over a set of images and labelled probes, merges
the boxes of all probes that share a class with weighted boxes fusion
(`merge_boxes`, the one merging step used for every model), and scores the
merged boxes against ground truth with per-class average precision at a
single IoU threshold (25% by default).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .image_prober import DetectedBox
from .image_prober import ImageInput
from .image_prober import ImageProber
from .image_prober import ProbeResult
from .image_prober import ResultKey
from .wbf import WBF
from .wbf import iou

# Merged predictions: {image key: {label: [boxes, best first]}}.
Predictions = dict[ResultKey, dict[str, list[DetectedBox]]]


@dataclass(frozen=True)
class GroundTruthBox:
    """One annotated object, in image pixels (x0, y0, x1, y1)."""

    label: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class Dataset:
    """A sample dataset's probes, images and ground truth, as `Evaluate` uses them."""

    name: str
    labels: list[str]
    probe_paths: list[str]
    probe_labels: dict[str, str]
    image_paths: list[str]
    ground_truth: dict[str, list[GroundTruthBox]]


class Evaluate:
    """Evaluate image probers' boxes: merge per class with WBF, then score by per-class AP."""

    def __init__(self, wbf: WBF, iou_threshold: float = 0.25) -> None:
        """Use `wbf` to merge boxes of same-class probes; count a match at IoU >= `iou_threshold`."""
        self.wbf = wbf
        self.iou_threshold = iou_threshold

    @staticmethod
    def load_dataset(dataset_dir: Path) -> Dataset:
        """Read a sample dataset's `probes.yaml` and `images.yaml` into a `Dataset` (boxes as xyxy)."""
        probes = yaml.safe_load((dataset_dir / "probes.yaml").read_text())
        images = yaml.safe_load((dataset_dir / "images.yaml").read_text())
        probe_paths = [str(dataset_dir / p["file"]) for p in probes["probes"]]
        image_paths = [str(dataset_dir / r["file"]) for r in images["images"]]
        ground_truth = {
            path: [GroundTruthBox(b["label"], x, y, x + w, y + h) for b in record["boxes"]
                   for x, y, w, h in [b["bbox_xywh"]]]
            for path, record in zip(image_paths, images["images"])
        }
        return Dataset(
            name=dataset_dir.name,
            labels=images["labels"],
            probe_paths=probe_paths,
            probe_labels={path: p["label"] for path, p in zip(probe_paths, probes["probes"])},
            image_paths=image_paths,
            ground_truth=ground_truth,
        )

    def predict(
        self, prober: ImageProber, images: Sequence[ImageInput], probes: Sequence[ImageInput],
        probe_labels: dict[ResultKey, str],
    ) -> Predictions:
        """Run `prober.image_probe` and merge each image's boxes per class with `merge_boxes`.

        `probe_labels` maps each probe's result key (its path as a string, or
        its index for in-memory images) to its class.
        """
        results = prober.image_probe(images, probes)
        return {image: self.merge_boxes(per_probe, probe_labels) for image, per_probe in results.items()}

    def merge_boxes(
        self, probe_results: dict[ResultKey, ProbeResult], probe_labels: dict[ResultKey, str]
    ) -> dict[str, list[DetectedBox]]:
        """Merge one image's boxes from all probes of each class into one box list per class.

        Each probe is one WBF source, so a box found by several probes of a
        class scores higher than one found by a single probe. This is the
        only merging step, applied the same way to every model.
        """
        by_label: dict[str, list[list[tuple[float, float, float, float, float]]]] = {}
        for key, result in probe_results.items():
            by_label.setdefault(probe_labels[key], []).append(
                [(b.x0, b.y0, b.x1, b.y1, b.score) for b in result.boxes])
        return {
            label: [DetectedBox(round(f.x0, 1), round(f.y0, 1), round(f.x1, 1), round(f.y1, 1), round(f.score, 4))
                    for f in self.wbf.fuse(sources)]
            for label, sources in by_label.items()
        }

    def average_precision(
        self, predictions: Predictions, ground_truth: dict[ResultKey, list[GroundTruthBox]], labels: Sequence[str]
    ) -> dict[str, float | None]:
        """Return box AP per class at `iou_threshold` (None for a class with no ground-truth boxes).

        All predicted boxes of a class, across all images, are ranked by
        score. Each is a true positive if it overlaps a not-yet-matched
        ground-truth box of that class in its image by at least the
        threshold (the best-overlapping one is taken), otherwise a false
        positive. AP is the area under the precision-recall curve with
        precision made monotonically decreasing (all-point interpolation,
        as in PASCAL VOC 2010+).
        """
        return {label: self._class_ap(predictions, ground_truth, label) for label in labels}

    def _class_ap(
        self, predictions: Predictions, ground_truth: dict[ResultKey, list[GroundTruthBox]], label: str
    ) -> float | None:
        """Average precision for one class (see `average_precision`)."""
        targets = {image: [(g.x0, g.y0, g.x1, g.y1) for g in boxes if g.label == label]
                   for image, boxes in ground_truth.items()}
        total = sum(len(t) for t in targets.values())
        if total == 0:
            return None
        ranked = sorted(((b.score, image, b) for image, per_label in predictions.items()
                         for b in per_label.get(label, [])), key=lambda item: -item[0])
        matched = {image: [False] * len(t) for image, t in targets.items()}
        hits: list[bool] = []
        for _, image, box in ranked:
            candidates = targets.get(image, [])
            overlaps = [iou((box.x0, box.y0, box.x1, box.y1), t) for t in candidates]
            best = max((i for i in range(len(candidates)) if not matched[image][i]),
                       key=lambda i: overlaps[i], default=None)
            hit = best is not None and overlaps[best] >= self.iou_threshold
            if hit:
                matched[image][best] = True
            hits.append(hit)
        # Precision envelope, then sum precision over each recall step.
        precisions, tp = [], 0
        for rank, hit in enumerate(hits, start=1):
            tp += hit
            precisions.append(tp / rank)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        return sum(p for p, hit in zip(precisions, hits) if hit) / total


def predictions_to_dict(predictions: Predictions) -> dict[str, Any]:
    """Convert merged predictions to plain lists for YAML: {image: {label: [[x0, y0, x1, y1, score], ...]}}."""
    return {str(image): {label: [[b.x0, b.y0, b.x1, b.y1, b.score] for b in boxes]
                         for label, boxes in per_label.items()}
            for image, per_label in predictions.items()}
