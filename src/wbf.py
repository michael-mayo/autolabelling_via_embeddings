"""Weighted boxes fusion (WBF) for merging overlapping boxes from several sources.

WBF (Solovyev, Wang & Gabruseva, 2021, "Weighted boxes fusion: Ensembling
boxes from different object detection models") merges boxes predicted by
several sources, e.g. several detectors, or several probes of the same class.
Unlike NMS, which keeps the best box of each overlapping group and discards
the rest, WBF averages the group's coordinates, weighted by score, so every
source contributes to the final box. A fused box's score is the group's
mean score, scaled down when fewer sources than available contributed, so
agreement between sources raises confidence.

The module is independent of the rest of the project: boxes are plain
`(x0, y0, x1, y1, score)` tuples in any consistent coordinate system, and
all boxes passed in one call are treated as the same class.
"""

from collections.abc import Sequence
from dataclasses import dataclass

# A box as (x0, y0, x1, y1, score).
BoxTuple = tuple[float, float, float, float, float]


@dataclass(frozen=True)
class FusedBox:
    """A box produced by fusing a group of overlapping input boxes."""

    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    count: int  # number of input boxes fused into this one


class WBF:
    """Weighted boxes fusion over boxes from several sources.

    `iou_threshold` is the overlap above which a box joins an existing
    group; `skip_box_threshold` drops low-scoring input boxes first.
    `conf_type` sets the fused score: `avg` (mean score, rescaled by
    min(group size, sources) / sources, as in the paper) or `max` (highest
    score in the group, no rescaling).
    """

    def __init__(self, iou_threshold: float = 0.55, skip_box_threshold: float = 0.0, conf_type: str = "avg") -> None:
        """Store the fusion settings."""
        if conf_type not in ("avg", "max"):
            raise ValueError(f"conf_type must be 'avg' or 'max', not {conf_type!r}")
        self.iou_threshold = iou_threshold
        self.skip_box_threshold = skip_box_threshold
        self.conf_type = conf_type

    def fuse(self, sources: Sequence[Sequence[BoxTuple]], weights: Sequence[float] | None = None) -> list[FusedBox]:
        """Fuse boxes from `sources` (one sequence of boxes per source) into a list sorted by score.

        Each source's scores are multiplied by its weight (default 1). Boxes
        are visited from highest to lowest score; each joins the first fused
        box it overlaps by more than `iou_threshold`, which is then
        recomputed as the score-weighted mean of its group. Scores are
        rescaled once all boxes are placed.
        """
        weights = list(weights) if weights is not None else [1.0] * len(sources)
        if len(weights) != len(sources):
            raise ValueError("weights must have one entry per source")
        boxes = sorted(
            ((x0, y0, x1, y1, score * w) for source, w in zip(sources, weights)
             for x0, y0, x1, y1, score in source if score >= self.skip_box_threshold),
            key=lambda b: -b[4],
        )
        groups: list[list[BoxTuple]] = []
        fused: list[BoxTuple] = []
        for box in boxes:
            match = self._best_match(fused, box)
            if match is None:
                groups.append([box])
                fused.append(box)
            else:
                groups[match].append(box)
                fused[match] = self._average(groups[match])
        total_weight = sum(weights)
        results = []
        for group, (x0, y0, x1, y1, _) in zip(groups, fused):
            scores = [b[4] for b in group]
            if self.conf_type == "max":
                score = max(scores)
            else:
                score = sum(scores) / len(scores) * min(len(group), total_weight) / total_weight
            results.append(FusedBox(x0, y0, x1, y1, score, len(group)))
        return sorted(results, key=lambda b: -b.score)

    def _best_match(self, fused: list[BoxTuple], box: BoxTuple) -> int | None:
        """Return the index of the fused box overlapping `box` most (above the threshold), or None."""
        best, best_iou = None, self.iou_threshold
        for i, candidate in enumerate(fused):
            overlap = iou(candidate, box)
            if overlap > best_iou:
                best, best_iou = i, overlap
        return best

    @staticmethod
    def _average(group: list[BoxTuple]) -> BoxTuple:
        """Return the score-weighted mean box of `group` (its score is the group's mean score)."""
        total = sum(b[4] for b in group)
        if total <= 0:
            total, wts = len(group), [1.0] * len(group)
        else:
            wts = [b[4] for b in group]
        coords = [sum(w * b[k] for w, b in zip(wts, group)) / total for k in range(4)]
        return (*coords, sum(b[4] for b in group) / len(group))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two boxes given as (x0, y0, x1, y1, ...)."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
