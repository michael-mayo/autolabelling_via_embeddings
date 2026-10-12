"""Select probes and images from an annotated source and write a sample dataset.

A sample dataset folder looks like:

    <name>/
      dataset.yaml         build settings (source, labels, sizes) - the only hand-written file
      probes.yaml          labels list + one entry per probe crop
      probes/probeN.jpg    labelled crops of single objects
      images/imageN.jpg    full, unlabelled images
      images.yaml          hidden per-image labels/boxes, for evaluation only

Probes are cropped from source images that are *not* used as dataset images,
so the probes never appear verbatim in the images they are searched for.
All images are resized to a fixed width, and probes are cropped from source
images resized the same way so that probes and images share a common scale.
"""

import io
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .sources import LOADERS
from .sources import AnnotatedImage
from .sources import Box
from .sources import fetch_image_bytes


# Name of the hand-written settings file inside each sample dataset folder.
DATASET_FILE = "dataset.yaml"
# Generated outputs that are replaced on every build.
GENERATED = ["probes", "images", "probes.yaml", "images.yaml"]


@dataclass(frozen=True)
class BuildSettings:
    """Selection and output settings read from one dataset's `dataset.yaml`."""

    num_probes: int
    num_images: int
    image_width: int
    negative_fraction: float
    probe_padding: float
    probe_min_side: int
    jpeg_quality: int
    seed: int

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "BuildSettings":
        """Build settings from a parsed `dataset.yaml`, ignoring the source-specific keys."""
        return cls(**{f: cfg[f] for f in cls.__dataclass_fields__})


def _split_evenly(total: int, labels: list[str]) -> dict[str, int]:
    """Split `total` across `labels` as evenly as possible, earlier labels first."""
    base, extra = divmod(total, len(labels))
    return {lab: base + (i < extra) for i, lab in enumerate(labels)}


def select_probes(
    pool: list[AnnotatedImage], labels: list[str], s: BuildSettings, rng: random.Random
) -> list[tuple[AnnotatedImage, Box]]:
    """Pick probe boxes, balanced across labels, each from a distinct source image.

    Only boxes whose shorter side is at least `probe_min_side` source pixels
    are eligible, so probes are large enough to be recognisable.
    """
    quota = _split_evenly(s.num_probes, labels)
    used: set[str] = set()
    chosen: list[tuple[AnnotatedImage, Box]] = []
    for label in labels:
        candidates = [
            (img, box)
            for img in pool
            for box in img.boxes
            if box.label == label and min(box.w, box.h) >= s.probe_min_side
        ]
        rng.shuffle(candidates)
        picked = 0
        for img, box in candidates:
            if picked == quota[label]:
                break
            if img.source_id in used:
                continue
            used.add(img.source_id)
            chosen.append((img, box))
            picked += 1
        if picked < quota[label]:
            raise ValueError(f"only {picked}/{quota[label]} probes available for label '{label}'")
    return chosen


def select_images(pool: list[AnnotatedImage], s: BuildSettings, rng: random.Random) -> list[AnnotatedImage]:
    """Pick full images covering as many distinct label combinations as possible.

    Positives are grouped by their exact label set (e.g. {dog}, {dog, bus})
    and drawn round-robin, always from the least-used combination that still
    has images. This mixes single- and multi-label images and stops a label
    that co-occurs with everything (e.g. caries) from dominating. Negatives
    (images with none of the labels) make up `negative_fraction` of the set
    when the source has any.
    """
    groups: dict[frozenset[str], list[AnnotatedImage]] = {}
    negatives = []
    for img in pool:
        if img.labels:
            groups.setdefault(frozenset(img.labels), []).append(img)
        else:
            negatives.append(img)
    for group in groups.values():
        rng.shuffle(group)
    rng.shuffle(negatives)
    chosen = negatives[: min(round(s.negative_fraction * s.num_images), len(negatives))]
    used = dict.fromkeys(groups, 0)
    # Sort combos deterministically, then shuffle, so ties are broken randomly but reproducibly.
    order = sorted(groups, key=lambda g: sorted(g))
    rng.shuffle(order)
    while len(chosen) < s.num_images:
        available = [g for g in order if groups[g]]
        if not available:
            raise ValueError(f"only {len(chosen)}/{s.num_images} images available")
        combo = min(available, key=lambda g: used[g])
        chosen.append(groups[combo].pop())
        used[combo] += 1
    rng.shuffle(chosen)
    return chosen


def _load_scaled(data: bytes, width: int) -> tuple[Image.Image, float]:
    """Decode image bytes and resize to `width`, keeping aspect ratio; return image and scale."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    scale = width / img.width
    size = (width, round(img.height * scale))
    return img.resize(size, Image.Resampling.LANCZOS), scale


def _crop_probe(img: Image.Image, box: Box, scale: float, padding: float) -> Image.Image:
    """Crop source-pixel `box` from `img` (resized by `scale`) with `padding` of context."""
    x, y, w, h = (v * scale for v in (box.x, box.y, box.w, box.h))
    left = max(0, int(x - w * padding))
    top = max(0, int(y - h * padding))
    right = min(img.width, int(x + w * (1 + padding)))
    bottom = min(img.height, int(y + h * (1 + padding)))
    return img.crop((left, top, right, bottom))


def _box_record(box: Box, scale: float = 1.0) -> dict[str, Any]:
    """Serialise a box for YAML output, scaling coordinates and rounding to 0.1 px."""
    return {"label": box.label, "bbox_xywh": [round(v * scale, 1) for v in (box.x, box.y, box.w, box.h)]}


def build_dataset(folder: Path, cache_dir: Path) -> None:
    """Build the sample dataset in `folder` from its `dataset.yaml`.

    Generated files (probes, images and their YAML indexes) are replaced;
    `dataset.yaml` and anything else in the folder is left untouched.
    """
    name = folder.name
    cfg = yaml.safe_load((folder / DATASET_FILE).read_text())
    s = BuildSettings.from_dict(cfg)
    rng = random.Random(s.seed)
    labels = [(cfg.get("label_map") or {}).get(n, n) for n in cfg["labels"]]
    pool = LOADERS[cfg["source"]](cfg, cache_dir)
    print(f"[{name}] {len(pool)} source images loaded")

    probes = select_probes(pool, labels, s, rng)
    probe_ids = {img.source_id for img, _ in probes}
    images = select_images([img for img in pool if img.source_id not in probe_ids], s, rng)

    for item in GENERATED:
        path = folder / item
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    (folder / "probes").mkdir()
    (folder / "images").mkdir()

    probe_records = []
    for i, (img, box) in enumerate(probes, start=1):
        rel = f"probes/probe{i}.jpg"
        scaled, scale = _load_scaled(fetch_image_bytes(img, cache_dir), s.image_width)
        _crop_probe(scaled, box, scale, s.probe_padding).save(folder / rel, quality=s.jpeg_quality)
        probe_records.append({
            "file": rel,
            "label": box.label,
            "source_id": img.source_id,
            "source_bbox_xywh": _box_record(box)["bbox_xywh"],
        })

    image_records = []
    for i, img in enumerate(images, start=1):
        rel = f"images/image{i}.jpg"
        scaled, scale = _load_scaled(fetch_image_bytes(img, cache_dir), s.image_width)
        scaled.save(folder / rel, quality=s.jpeg_quality)
        image_records.append({
            "file": rel,
            "source_id": img.source_id,
            "labels": sorted(img.labels),
            "boxes": [_box_record(b, scale) for b in img.boxes],
        })

    provenance = {"source": cfg["dataset"], "license": cfg["license"]}
    _write_yaml(folder / "probes.yaml", {**provenance, "labels": labels, "probes": probe_records})
    _write_yaml(folder / "images.yaml", {
        **provenance,
        "note": "Hidden image labels for evaluation only; not an input to labelling.",
        "labels": labels,
        "images": image_records,
    })
    _print_summary(name, labels, probe_records, image_records)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write `data` as block-style YAML, preserving key order."""
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=120))


def _print_summary(
    name: str, labels: list[str], probes: list[dict[str, Any]], images: list[dict[str, Any]]
) -> None:
    """Print per-label probe/image counts for a built dataset."""
    multi = sum(len(r["labels"]) > 1 for r in images)
    empty = sum(not r["labels"] for r in images)
    print(f"[{name}] {len(probes)} probes, {len(images)} images ({multi} multi-label, {empty} negative)")
    for lab in labels:
        n_p = sum(r["label"] == lab for r in probes)
        n_i = sum(lab in r["labels"] for r in images)
        print(f"    {lab:<20} probes={n_p:<3} images={n_i}")


def find_datasets(datasets_dir: Path) -> list[Path]:
    """Return every sub-folder of `datasets_dir` that contains a `dataset.yaml`, sorted by name."""
    return sorted(p for p in datasets_dir.iterdir() if (p / DATASET_FILE).exists())


def build_all(config: dict[str, Any], only: list[str] | None = None) -> None:
    """Build every sample dataset found under `config["sample_data"]["datasets_dir"]`.

    `only` restricts the build to the named dataset folders. Downloads are
    cached per dataset under `<general.weights_dir>/sample_data_cache/`.
    """
    datasets_dir = Path(config["sample_data"]["datasets_dir"])
    cache_root = Path(config["general"]["weights_dir"]) / "sample_data_cache"
    for folder in find_datasets(datasets_dir):
        if only and folder.name not in only:
            continue
        build_dataset(folder, cache_root / folder.name)
