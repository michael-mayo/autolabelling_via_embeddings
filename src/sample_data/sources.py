"""Fetchers that pull annotated detection images from public Hugging Face datasets.

Each source returns a list of `AnnotatedImage` records holding the image's
bounding boxes (already mapped to the sample dataset's label names) and a way
to fetch the raw JPEG bytes. Image bytes and annotation pages are cached on
disk so that rebuilding the sample datasets does not re-download anything.
"""

import hashlib
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import requests

# Hugging Face datasets-server endpoint that serves dataset rows as JSON.
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
# Maximum rows the datasets-server returns per request.
HF_ROWS_PAGE = 100
# Seconds to wait on any single HTTP request.
HTTP_TIMEOUT = 60
# Parallel HTTP requests used when fetching many small annotation files.
HTTP_WORKERS = 16


@dataclass(frozen=True)
class Box:
    """A labelled bounding box in pixel coordinates (x, y, width, height)."""

    label: str
    x: float
    y: float
    w: float
    h: float


@dataclass
class AnnotatedImage:
    """An image from a source dataset together with its target-label boxes."""

    source_id: str
    url: str
    boxes: list[Box] = field(default_factory=list)

    @property
    def labels(self) -> set[str]:
        """Return the distinct labels present in the image."""
        return {b.label for b in self.boxes}


def _get_json(url: str, params: dict[str, Any] | None, cache_file: Path) -> Any:
    """Fetch JSON from `url`, reusing `cache_file` if it already exists."""
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text)
    return resp.json()


def fetch_image_bytes(image: AnnotatedImage, cache_dir: Path) -> bytes:
    """Return the raw encoded image bytes for `image`, downloading into `cache_dir` once."""
    # Hash the source id so ids containing slashes are safe filenames.
    name = hashlib.sha1(image.source_id.encode()).hexdigest()[:16] + ".img"
    path = cache_dir / "images" / name
    if not path.exists():
        resp = requests.get(image.url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    return path.read_bytes()


def _map_labels(raw_names: list[str], cfg: dict[str, Any]) -> dict[int, str]:
    """Map source category ids to output label names, keeping only target labels.

    `cfg["labels"]` lists the source category names to keep and
    `cfg.get("label_map")` optionally renames them for the sample dataset.
    """
    rename = cfg.get("label_map") or {}
    keep = set(cfg["labels"])
    return {i: rename.get(n, n) for i, n in enumerate(raw_names) if n in keep}


def load_hf_rows(cfg: dict[str, Any], cache_dir: Path) -> list[AnnotatedImage]:
    """Load images and boxes from a dataset served by the HF datasets-server rows API.

    Expects rows with an `image` field and an `objects` dict holding COCO-style
    `bbox` lists (x, y, w, h) and a ClassLabel field named by `cfg["label_field"]`.
    Crowd annotations (`iscrowd`) are skipped.
    """
    label_field = cfg["label_field"]
    images: list[AnnotatedImage] = []
    id_to_label: dict[int, str] | None = None
    for offset in range(0, cfg["max_rows"], HF_ROWS_PAGE):
        params = {
            "dataset": cfg["dataset"],
            "config": cfg["config"],
            "split": cfg["split"],
            "offset": offset,
            "length": HF_ROWS_PAGE,
        }
        page = _get_json(HF_ROWS_URL, params, cache_dir / "rows" / f"{offset:06d}.json")
        if id_to_label is None:
            objects = next(f for f in page["features"] if f["name"] == "objects")
            raw_names = objects["type"][label_field]["feature"]["names"]
            id_to_label = _map_labels(raw_names, cfg)
        for r in page["rows"]:
            row = r["row"]
            objs = row["objects"]
            crowd = objs.get("iscrowd") or [False] * len(objs["bbox"])
            boxes = [
                Box(id_to_label[c], *bbox)
                for bbox, c, is_crowd in zip(objs["bbox"], objs[label_field], crowd)
                if c in id_to_label and not is_crowd
            ]
            # Prefix with dataset name so ids stay unique across sources.
            images.append(AnnotatedImage(f"{cfg['dataset']}:{row['image_id']}", row["image"]["src"], boxes))
        if len(page["rows"]) < HF_ROWS_PAGE:
            break
    return images


def load_rdd2022(cfg: dict[str, Any], cache_dir: Path) -> list[AnnotatedImage]:
    """Load images and boxes from the RDD2022 road-damage repo on the HF hub.

    Annotations come from each shard's `metadata.jsonl` (COCO-style x, y, w, h
    boxes with integer category ids); images are fetched via the hub's
    `resolve` URLs.
    """
    id_to_label = _map_labels(cfg["category_names"], cfg)
    base = f"https://huggingface.co/datasets/{cfg['dataset']}/resolve/main/data/images"
    images: list[AnnotatedImage] = []
    for shard in cfg["shards"]:
        shard_url = f"{base}/{shard}"
        cache_file = cache_dir / "metadata" / f"{shard.replace('/', '_')}.jsonl"
        if not cache_file.exists():
            resp = requests.get(f"{shard_url}/metadata.jsonl", timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(resp.text)
        for line in cache_file.read_text().splitlines():
            row = json.loads(line)
            objs = row["objects"]
            boxes = [
                Box(id_to_label[c], *bbox)
                for bbox, c in zip(objs["bbox"], objs["categories"])
                if c in id_to_label
            ]
            images.append(AnnotatedImage(f"{shard}/{row['file_name']}", f"{shard_url}/{row['file_name']}", boxes))
    return images


def _list_hub_files(dataset: str, folder: str) -> list[str]:
    """List file paths in `folder` of a HF hub dataset repo, following pagination."""
    url = f"https://huggingface.co/api/datasets/{dataset}/tree/main/{folder}?limit=1000"
    paths: list[str] = []
    while url:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        paths += [x["path"] for x in resp.json() if x["type"] == "file"]
        url = resp.links.get("next", {}).get("url")
    return paths


def _png_size(url: str) -> tuple[int, int]:
    """Read a PNG's (width, height) from its IHDR header via an HTTP range request."""
    resp = requests.get(url, headers={"Range": "bytes=0-23"}, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return struct.unpack(">II", resp.content[16:24])


def load_yolo_hub(cfg: dict[str, Any], cache_dir: Path) -> list[AnnotatedImage]:
    """Load images and boxes from a YOLO-format dataset repo on the HF hub.

    Images in `cfg["image_dir"]` whose names start with `cfg["file_prefix"]`
    are used, with labels from the matching `.txt` in `cfg["label_dir"]`.
    YOLO boxes are normalised, so each PNG's pixel size is read from its header
    (without downloading the whole image) to convert them to pixels. All
    annotations are cached in one JSON file.
    """
    id_to_label = _map_labels(cfg["category_names"], cfg)
    base = f"https://huggingface.co/datasets/{cfg['dataset']}/resolve/main"
    cache_file = cache_dir / "annotations.json"
    if cache_file.exists():
        records = json.loads(cache_file.read_text())
    else:
        files = [
            p for p in _list_hub_files(cfg["dataset"], cfg["image_dir"])
            if Path(p).name.startswith(cfg["file_prefix"])
        ]

        def fetch(path: str) -> dict[str, Any]:
            """Fetch the label text and pixel size for one image."""
            label_path = f"{cfg['label_dir']}/{Path(path).stem}.txt"
            resp = requests.get(f"{base}/{label_path}", timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return {"path": path, "size": _png_size(f"{base}/{path}"), "labels": resp.text}

        with ThreadPoolExecutor(HTTP_WORKERS) as pool:
            records = list(pool.map(fetch, files))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(records))

    images: list[AnnotatedImage] = []
    for rec in records:
        width, height = rec["size"]
        boxes = []
        for line in rec["labels"].splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            if int(c) in id_to_label:
                bw, bh = float(w) * width, float(h) * height
                boxes.append(Box(id_to_label[int(c)], float(cx) * width - bw / 2, float(cy) * height - bh / 2, bw, bh))
        images.append(AnnotatedImage(rec["path"], f"{base}/{rec['path']}", boxes))
    return images


# Registry of source loaders, keyed by the `source` value in config.yaml.
LOADERS = {
    "hf_rows": load_hf_rows,
    "rdd2022": load_rdd2022,
    "yolo_hub": load_yolo_hub,
}
