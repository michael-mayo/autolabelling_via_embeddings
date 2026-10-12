"""Select probes and images from an annotated source and write a sample dataset.

A sample dataset folder looks like:

    <name>/
      probes.yaml          labels list + one entry per probe crop
      probes/probeN.jpg    labelled crops of single objects
      images/imageN.jpg    full, unlabelled images
      images.yaml          hidden per-image labels/boxes, for evaluation only

Probes are cropped from source images that are *not* used as dataset images,
so the probes never appear verbatim in the images they are searched for.
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


@dataclass(frozen=True)
class BuildSettings:
    """Settings shared by all sample datasets (the `sample_data` config section)."""

    output_dir: Path
    cache_dir: Path
    num_probes: int
    num_images: int
    negative_fraction: float
    probe_padding: float
    probe_min_side: int
    jpeg_quality: int
    seed: int


def _split_evenly(total: int, labels: list[str]) -> dict[str, int]:
    """Split `total` across `labels` as evenly as possible, earlier labels first."""
    base, extra = divmod(total, len(labels))
    return {lab: base + (i < extra) for i, lab in enumerate(labels)}


def select_probes(
    pool: list[AnnotatedImage], labels: list[str], s: BuildSettings, min_side: int, rng: random.Random
) -> list[tuple[AnnotatedImage, Box]]:
    """Pick probe boxes, balanced across labels, each from a distinct source image.

    Only boxes whose shorter side is at least `min_side` pixels are eligible,
    so probes are large enough to be recognisable.
    """
    quota = _split_evenly(s.num_probes, labels)
    used: set[str] = set()
    chosen: list[tuple[AnnotatedImage, Box]] = []
    for label in labels:
        candidates = [
            (img, box)
            for img in pool
            for box in img.boxes
            if box.label == label and min(box.w, box.h) >= min_side
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


def _crop_probe(data: bytes, box: Box, padding: float) -> Image.Image:
    """Crop `box` out of a JPEG with `padding` (fraction of box size) of context."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    pad_x, pad_y = box.w * padding, box.h * padding
    left = max(0, int(box.x - pad_x))
    top = max(0, int(box.y - pad_y))
    right = min(img.width, int(box.x + box.w + pad_x))
    bottom = min(img.height, int(box.y + box.h + pad_y))
    return img.crop((left, top, right, bottom))


def _write_image(data: bytes, path: Path, quality: int, max_side: int | None) -> float:
    """Write image bytes to `path` as JPEG, downscaling so neither side exceeds `max_side`.

    JPEG input that needs no resizing is copied as-is to avoid re-compression.
    Returns the scale factor applied, so box coordinates can be adjusted.
    """
    img = Image.open(io.BytesIO(data))
    scale = min(1.0, max_side / max(img.size)) if max_side else 1.0
    if scale == 1.0 and img.format == "JPEG":
        path.write_bytes(data)
        return scale
    img = img.convert("RGB")
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    img.save(path, quality=quality)
    return scale


def _box_record(box: Box, scale: float = 1.0) -> dict[str, Any]:
    """Serialise a box for YAML output, scaling coordinates and rounding to 0.1 px."""
    return {"label": box.label, "bbox_xywh": [round(v * scale, 1) for v in (box.x, box.y, box.w, box.h)]}


def build_dataset(name: str, cfg: dict[str, Any], s: BuildSettings) -> None:
    """Build one sample dataset `name` from its config section and write it to disk.

    Any existing folder for this dataset is replaced.
    """
    rng = random.Random(s.seed)
    cache_dir = s.cache_dir / name
    labels = [(cfg.get("label_map") or {}).get(n, n) for n in cfg["labels"]]
    pool = LOADERS[cfg["source"]](cfg, cache_dir)
    print(f"[{name}] {len(pool)} source images loaded")

    # Datasets with small objects (e.g. platelets) can override the probe size floor.
    probes = select_probes(pool, labels, s, cfg.get("probe_min_side", s.probe_min_side), rng)
    probe_ids = {img.source_id for img, _ in probes}
    images = select_images([img for img in pool if img.source_id not in probe_ids], s, rng)

    out = s.output_dir / name
    if out.exists():
        shutil.rmtree(out)
    (out / "probes").mkdir(parents=True)
    (out / "images").mkdir()

    probe_records = []
    for i, (img, box) in enumerate(probes, start=1):
        rel = f"probes/probe{i}.jpg"
        crop = _crop_probe(fetch_image_bytes(img, cache_dir), box, s.probe_padding)
        crop.save(out / rel, quality=s.jpeg_quality)
        probe_records.append({"file": rel, "label": box.label, "source_id": img.source_id, **_box_record(box)})

    image_records = []
    for i, img in enumerate(images, start=1):
        rel = f"images/image{i}.jpg"
        scale = _write_image(fetch_image_bytes(img, cache_dir), out / rel, s.jpeg_quality, cfg.get("max_image_side"))
        image_records.append({
            "file": rel,
            "source_id": img.source_id,
            "labels": sorted(img.labels),
            "boxes": [_box_record(b, scale) for b in img.boxes],
        })

    provenance = {"source": cfg["dataset"], "license": cfg["license"]}
    _write_yaml(out / "probes.yaml", {**provenance, "labels": labels, "probes": probe_records})
    _write_yaml(out / "images.yaml", {
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


def build_all(config: dict[str, Any], only: list[str] | None = None) -> None:
    """Build every sample dataset in `config["sample_data"]`, or just those in `only`."""
    section = config["sample_data"]
    s = BuildSettings(
        output_dir=Path(section["output_dir"]),
        cache_dir=Path(section["cache_dir"]),
        num_probes=section["num_probes"],
        num_images=section["num_images"],
        negative_fraction=section["negative_fraction"],
        probe_padding=section["probe_padding"],
        probe_min_side=section["probe_min_side"],
        jpeg_quality=section["jpeg_quality"],
        seed=config["general"]["random_seed"],
    )
    for name, cfg in section["datasets"].items():
        if only and name not in only:
            continue
        build_dataset(name, cfg, s)
