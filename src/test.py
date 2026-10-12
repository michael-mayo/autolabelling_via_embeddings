"""Render each sample dataset's images with their ground-truth boxes into a PDF.

For every dataset folder under the sample datasets directory, this reads
`images.yaml`, draws each labelled box on its image, adds a header listing the
image's labels, and writes one PDF per dataset (one image per page) to the
scratch directory.
"""

import argparse
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Distinct, colour-blind-friendly colours assigned to labels in order.
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]


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


def render_dataset(dataset_dir: Path, out_dir: Path, cfg: dict[str, Any]) -> Path:
    """Write `<dataset name>.pdf` to `out_dir` with one annotated image per page."""
    meta = yaml.safe_load((dataset_dir / "images.yaml").read_text())
    colours = _label_colours(meta["labels"])
    pages = [render_page(dataset_dir / r["file"], r, colours, cfg) for r in meta["images"]]
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
    for dataset_dir in sorted(p for p in datasets_dir.iterdir() if (p / "images.yaml").exists()):
        out = render_dataset(dataset_dir, out_dir, cfg)
        print(f"[{dataset_dir.name}] wrote {out}")


if __name__ == "__main__":
    main()
