"""Render each sample dataset's images with their ground-truth boxes into a PDF.

For every dataset folder under the sample datasets directory, this reads
`images.yaml`, draws each labelled box on its image, adds a header listing the
image's labels, and writes one PDF per dataset (one image per page) to the
scratch directory. Each PDF opens with a cover page summarising every sample
dataset side by side, with the PDF's own dataset highlighted.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Distinct, colour-blind-friendly colours assigned to labels in order.
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]


@dataclass(frozen=True)
class ClassStats:
    """Ground-truth statistics for one label within a dataset."""

    label: str
    boxes: int        # boxes with this label
    images: int       # support: images containing at least one such box
    probes: int       # probe crops with this label


@dataclass(frozen=True)
class DatasetStats:
    """Ground-truth statistics for one sample dataset."""

    name: str
    source: str
    images: int
    boxes: int
    background: int   # images with none of the labels
    multi_label: int  # images with two or more distinct labels
    probes: int
    classes: list[ClassStats]


def dataset_stats(dataset_dir: Path) -> DatasetStats:
    """Compute summary statistics for one dataset from its `images.yaml` and `probes.yaml`."""
    meta = yaml.safe_load((dataset_dir / "images.yaml").read_text())
    probes = yaml.safe_load((dataset_dir / "probes.yaml").read_text())["probes"]
    records = meta["images"]
    classes = []
    for lab in meta["labels"]:
        classes.append(ClassStats(
            label=lab,
            boxes=sum(b["label"] == lab for r in records for b in r["boxes"]),
            images=sum(lab in r["labels"] for r in records),
            probes=sum(p["label"] == lab for p in probes),
        ))
    return DatasetStats(
        name=dataset_dir.name,
        source=meta["source"],
        images=len(records),
        boxes=sum(len(r["boxes"]) for r in records),
        background=sum(not r["labels"] for r in records),
        multi_label=sum(len(r["labels"]) > 1 for r in records),
        probes=len(probes),
        classes=classes,
    )


def render_cover(all_stats: list[DatasetStats], current: str, cfg: dict[str, Any]) -> Image.Image:
    """Return a cover page summarising every dataset in its own column.

    The column for `current` (the PDF's own dataset) is shaded. Each column
    gives dataset totals and a per-class table of boxes, support (images
    containing the class), image coverage (support as a % of images) and
    probe count.
    """
    page_w, page_h = cfg["page_size"]
    fs = cfg["cover_font_size"]
    line = round(fs * 1.35)
    font = ImageFont.load_default(fs)
    bold = ImageFont.load_default(round(fs * 1.2))
    title = ImageFont.load_default(round(fs * 1.4))
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    draw.text((20, 12), "Sample datasets - ground-truth summary", fill="black", font=title)

    col_w = page_w // len(all_stats)
    top = 12 + round(fs * 1.4) + line
    headers = ["boxes", "support", "% images", "probes"]
    num_w = (col_w - 40) * 0.16   # width of each numeric column; class name takes the rest
    for i, st in enumerate(all_stats):
        x0 = i * col_w + 20
        if st.name == current:
            draw.rectangle([i * col_w + 6, top - 8, (i + 1) * col_w - 6, page_h - 2 * line], fill="#EEF4FA")
        y = top
        draw.text((x0, y), st.name, fill="black", font=bold)
        y += line + 2
        draw.text((x0, y), st.source, fill="#555555", font=font)
        y += line
        draw.text((x0, y), f"images {st.images}   boxes {st.boxes}   probes {st.probes}", fill="black", font=font)
        y += line
        draw.text((x0, y), f"background {st.background}   multi-label {st.multi_label}   "
                  f"boxes/image {st.boxes / st.images:.1f}", fill="black", font=font)
        y += round(line * 1.5)

        # Per-class table: class name left-aligned, numbers right-aligned in fixed columns.
        right_edges = [x0 + (col_w - 40) - num_w * (len(headers) - 1 - k) for k in range(len(headers))]
        rows = [(c.label, [c.boxes, c.images, f"{100 * c.images / st.images:.0f}%", c.probes])
                for c in st.classes]
        draw.text((x0, y), "class", fill="#555555", font=font)
        for edge, h in zip(right_edges, headers):
            draw.text((edge - draw.textlength(h, font=font), y), h, fill="#555555", font=font)
        y += line
        draw.line([x0, y - 4, x0 + col_w - 40, y - 4], fill="#999999", width=1)
        for label, values in rows:
            draw.text((x0, y), label, fill="black", font=font)
            for edge, v in zip(right_edges, values):
                text = str(v)
                draw.text((edge - draw.textlength(text, font=font), y), text, fill="black", font=font)
            y += line

    note = ("support = images containing the class; % images = support / total images; "
            "background = images with no labels")
    draw.text((20, page_h - line - 6), note, fill="#555555", font=ImageFont.load_default(round(fs * 0.8)))
    return page


def _label_colours(labels: list[str]) -> dict[str, str]:
    """Assign each label a palette colour, cycling if there are more labels than colours."""
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}


def render_page(
    image_path: Path, record: dict[str, Any], colours: dict[str, str], cfg: dict[str, Any]
) -> Image.Image:
    """Return one PDF page: the image with boxes drawn, under a header naming its labels.

    Every page has the same fixed size, `cfg["page_size"]` (width, height in
    px). PDF viewers usually zoom to the page width, so a wide, short page
    takes up a predictable share of the screen whatever the image's shape.
    The image is scaled to fit below the header and centred horizontally.
    """
    page_w, page_h = cfg["page_size"]
    header_h = cfg["font_size"] * 3
    img = Image.open(image_path).convert("RGB")
    scale = min(page_w / img.width, (page_h - header_h) / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    font = ImageFont.load_default(cfg["font_size"])
    draw = ImageDraw.Draw(img)
    for box in record["boxes"]:
        x, y, w, h = (v * scale for v in box["bbox_xywh"])
        colour = colours[box["label"]]
        draw.rectangle([x, y, x + w, y + h], outline=colour, width=cfg["line_width"])
        # Label tag sits above the box, or inside it when the box touches the top edge.
        tag = draw.textbbox((0, 0), box["label"], font=font)
        tag_h = tag[3] - tag[1] + 4
        ty = y - tag_h if y >= tag_h else y
        draw.rectangle([x, ty, x + tag[2] + 6, ty + tag_h], fill=colour)
        draw.text((x + 3, ty + 1), box["label"], fill="black", font=font)

    page = Image.new("RGB", (page_w, page_h), "white")
    page.paste(img, ((page_w - img.width) // 2, header_h))
    hdraw = ImageDraw.Draw(page)
    labels = ", ".join(record["labels"]) or "(no labels)"
    hdraw.text((10, cfg["font_size"] // 2), f"{record['file']}    labels: {labels}", fill="black", font=font)
    hdraw.text((10, cfg["font_size"] * 3 // 2), f"source: {record['source_id']}", fill="#555555", font=font)
    return page


def render_dataset(dataset_dir: Path, out_dir: Path, all_stats: list[DatasetStats], cfg: dict[str, Any]) -> Path:
    """Write `<dataset name><suffix>.pdf` to `out_dir`: a summary cover, then one annotated image per page."""
    meta = yaml.safe_load((dataset_dir / "images.yaml").read_text())
    colours = _label_colours(meta["labels"])
    pages = [render_cover(all_stats, dataset_dir.name, cfg)]
    pages += [render_page(dataset_dir / r["file"], r, colours, cfg) for r in meta["images"]]
    out = out_dir / f"{dataset_dir.name}{cfg['pdf_suffix']}.pdf"
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=cfg["dpi"])
    return out


def main() -> None:
    """Parse arguments and render a ground-truth PDF for every sample dataset."""
    parser = argparse.ArgumentParser(description="Render sample dataset images with ground-truth boxes to PDFs.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    cfg = config["test"]
    out_dir = Path(config["general"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = Path(config["sample_data"]["datasets_dir"])
    dataset_dirs = sorted(p for p in datasets_dir.iterdir() if (p / "images.yaml").exists())
    all_stats = [dataset_stats(d) for d in dataset_dirs]
    for dataset_dir in dataset_dirs:
        out = render_dataset(dataset_dir, out_dir, all_stats, cfg)
        print(f"[{dataset_dir.name}] wrote {out}")


if __name__ == "__main__":
    main()
