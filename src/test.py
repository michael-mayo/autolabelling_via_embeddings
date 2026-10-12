"""Evaluate every image prober's boxes on every sample dataset, and render PDF reports.

For each model in `test.models` and each sample dataset, this runs the
prober through `Evaluate`, merging the boxes of same-class probes with WBF,
and scores the merged boxes with per-class box AP (IoU >= `evaluate.iou_threshold`).
It then:

- scores per-image classification: an image is predicted to contain a class
  when one of its merged boxes of that class scores above a threshold T,
  chosen per class to maximise F1 (precision, recall, F1 and T are reported);
- prints, at the end, a box-AP table per dataset (rows: models; columns:
  classes and mean AP) and a models x datasets summary, then the same for
  classification (rows: model x class; columns: P, R, F1, T), with the best
  values highlighted;
- first deletes the previous run's outputs (below), so a partial run reports
  only its own results;
- writes `<output_dir>/evaluation.yaml` with the AP table and every model's
  merged (final) predictions;
- writes `<output_dir>/<dataset><pdf_suffix>.pdf`: a cover summarising every
  dataset, a box-AP page, a classification page, then the ground truth drawn
  on each image;
- writes `<output_dir>/<dataset>_<model>.pdf`: the box-AP and classification
  pages with the model's row highlighted, then each image with its ground truth (thin) and
  the model's top merged predictions per class (thick, with scores).
"""

import argparse
import gc
import importlib
import math
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from rich import box
from rich.console import Console
from rich.table import Table

from .evaluate import ClassificationStats
from .evaluate import Dataset
from .evaluate import Evaluate
from .evaluate import Predictions
from .evaluate import predictions_to_dict
from .image_prober import ImageProber
from .wbf import WBF

# Distinct, colour-blind-friendly colours assigned to labels in order.
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
# Model name (as used in config.yaml `test.models`) -> (module in this package, class name).
MODELS = {
    "owlv2": ("image_prober_owlv2", "ImageProber_OwlV2"),
    "sam3": ("image_prober_sam3", "ImageProber_SAM3"),
    "dinov3": ("image_prober_dinov3", "ImageProber_DINOv3"),
    "dinov2": ("image_prober_dinov2", "ImageProber_DINOv2"),
    "yoloe": ("image_prober_yoloe", "ImageProber_YOLOE"),
    "embeddinggemma2": ("image_prober_embeddinggemma2", "ImageProber_EmbeddingGemma2"),
}
# Box AP per model, dataset and class: {model: {dataset: {label: AP or None}}}.
APTable = dict[str, dict[str, dict[str, float | None]]]
# Per-image classification stats per model, dataset and class.
ClsTable = dict[str, dict[str, dict[str, ClassificationStats | None]]]


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


# ----- evaluation -------------------------------------------------------------------


def load_prober(name: str, config: dict[str, Any], box_threshold: float | None = None) -> ImageProber:
    """Import and build the prober registered as `name` in MODELS from `config`.

    A non-None `box_threshold` replaces the model's own `box_threshold` for
    this run; the model's config section is otherwise used unchanged.
    """
    module_name, class_name = MODELS[name]
    cls = getattr(importlib.import_module(f".{module_name}", package=__package__), class_name)
    if box_threshold is not None:
        config = {**config, cls.config_section: {**config[cls.config_section], "box_threshold": box_threshold}}
    return cls.from_config(config)


def mean_ap(per_class: dict[str, float | None]) -> float | None:
    """Mean of the defined per-class APs (None if no class has ground truth)."""
    values = [v for v in per_class.values() if v is not None]
    return sum(values) / len(values) if values else None


def mean_stat(per_class: dict[str, ClassificationStats | None], field: str) -> float | None:
    """Mean of one ClassificationStats field over classes where it is defined (None if nowhere)."""
    values = [getattr(c, field) for c in per_class.values() if c is not None and getattr(c, field) is not None]
    return sum(values) / len(values) if values else None


def mean_f1(per_class: dict[str, ClassificationStats | None]) -> float | None:
    """Mean threshold-tuned F1 over classes that occur in the dataset (None if none do)."""
    return mean_stat(per_class, "f1")


def _fmt(value: float | None) -> str:
    """Format an AP for display ('-' when undefined)."""
    return "-" if value is None or math.isnan(value) else f"{value:.3f}"


def _ap_cells(values: list[float | None]) -> list[str]:
    """Format a column of APs as rich markup, with the best value(s) in bold green."""
    defined = [v for v in values if v is not None and not math.isnan(v)]
    best = max(defined, default=None)
    return [f"[bold green]{_fmt(v)}[/]" if v is not None and best and v == best else _fmt(v) for v in values]


def print_tables(ap: APTable, datasets: list[Dataset], iou_threshold: float) -> None:
    """Print a box-AP table per dataset (models x classes + mean), then a models x datasets summary.

    The best value in each column is highlighted.
    """
    # Piped output has no terminal width; use a wide one so class names are not truncated.
    console = Console(width=None if sys.stdout.isatty() else 200)
    console.print()
    console.rule(f"[bold]Box AP @ IoU {iou_threshold:g}[/]  (same-class probe boxes merged with WBF)")
    models = list(ap)
    for ds in datasets:
        table = Table(title=f"[bold]{ds.name}[/]", box=box.ROUNDED, header_style="bold", title_justify="left")
        table.add_column("model")
        columns = []
        for label in ds.labels:
            table.add_column(label, justify="right")
            columns.append([ap[m][ds.name][label] for m in models])
        table.add_column("mean", justify="right", style="bold")
        columns.append([mean_ap(ap[m][ds.name]) for m in models])
        cells = [_ap_cells(col) for col in columns]
        for i, model in enumerate(models):
            table.add_row(model, *(col[i] for col in cells))
        console.print(table)

    summary = Table(title="[bold]Mean box AP per dataset[/]", box=box.ROUNDED, header_style="bold", title_justify="left")
    summary.add_column("model")
    columns = []
    for ds in datasets:
        summary.add_column(ds.name, justify="right")
        columns.append([mean_ap(ap[m][ds.name]) for m in models])
    cells = [_ap_cells(col) for col in columns]
    for i, model in enumerate(models):
        summary.add_row(model, *(col[i] for col in cells))
    console.print(summary)


def print_classification_tables(cls: ClsTable, datasets: list[Dataset]) -> None:
    """Print per-image classification stats per dataset (rows: model x class), then a summary of means.

    Each row gives the every-image-positive F1 baseline, the threshold-tuned
    F1 (with its T, precision and recall) and the threshold-free AUROC. The
    best tuned F1 and AUROC per class are highlighted; tuned F1 is dimmed
    when it does not beat the baseline.
    """
    console = Console(width=None if sys.stdout.isatty() else 200)
    console.print()
    console.rule("[bold]Per-image classification[/]  (image score = best merged box of the class; "
                 "tuned F1 at the F1-maximising T; AUROC is threshold-free)")
    models = list(cls)
    for ds in datasets:
        table = Table(title=f"[bold]{ds.name}[/]", box=box.ROUNDED, header_style="bold", title_justify="left")
        for name, justify in [("model", "left"), ("class", "left"), ("images", "right"),
                              ("baseline F1", "right"), ("tuned F1", "right"), ("AUROC", "right"),
                              ("T", "right"), ("precision", "right"), ("recall", "right")]:
            table.add_column(name, justify=justify)
        stats = {label: [cls[m][ds.name][label] for m in models] for label in ds.labels}
        best_f1 = {label: max((c.f1 for c in col if c), default=None) for label, col in stats.items()}
        best_auc = {label: max((c.auroc for c in col if c and c.auroc is not None), default=None)
                    for label, col in stats.items()}
        for i, model in enumerate(models):
            if i:
                table.add_section()
            for label in ds.labels:
                c = cls[model][ds.name][label]
                if c is None:
                    table.add_row(model, label, "0", *["-"] * 6)
                    continue
                f1 = f"{c.f1:.3f}"
                if c.f1 <= c.baseline_f1 + 1e-9:
                    f1 = f"[dim]{f1}[/]"
                elif c.f1 == best_f1[label]:
                    f1 = f"[bold green]{f1}[/]"
                auc = _fmt(c.auroc)
                if c.auroc is not None and c.auroc == best_auc[label]:
                    auc = f"[bold green]{auc}[/]"
                threshold = "-" if c.threshold is None else f"{c.threshold:.4f}"
                table.add_row(model, label, str(c.positives), f"{c.baseline_f1:.3f}", f1, auc, threshold,
                              f"{c.precision:.3f}", f"{c.recall:.3f}")
        console.print(table)

    summary = Table(title="[bold]Per-image classification: means over classes[/]", box=box.ROUNDED,
                    header_style="bold", title_justify="left")
    summary.add_column("model")
    columns = []
    for ds in datasets:
        for field, name in [("f1", "tuned F1"), ("auroc", "AUROC")]:
            summary.add_column(f"{ds.name}\n{name}", justify="right")
            columns.append([mean_stat(cls[m][ds.name], field) for m in models])
    cells = [_ap_cells(col) for col in columns]
    # The baseline depends only on the ground truth, so it is one row; its AUROC is 0.5 by definition.
    baseline = []
    for ds in datasets:
        first = cls[models[0]][ds.name]
        baseline += [_fmt(mean_stat(first, "baseline_f1")), "0.500"]
    summary.add_row("[italic]every image positive[/]", *baseline)
    summary.add_section()
    for i, model in enumerate(models):
        summary.add_row(model, *(col[i] for col in cells))
    console.print(summary)


# ----- PDF rendering ----------------------------------------------------------------


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


def render_ap_page(
    ap: APTable, dataset: Dataset, iou_threshold: float, highlight: str | None, cfg: dict[str, Any]
) -> Image.Image:
    """Return a page with one dataset's box-AP table: a row per model, a column per class plus mean.

    The `highlight` model's row (the report's own model) is shaded.
    """
    page_w, page_h = cfg["page_size"]
    fs = cfg["cover_font_size"]
    line = round(fs * 1.6)
    font = ImageFont.load_default(fs)
    title = ImageFont.load_default(round(fs * 1.4))
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    draw.text((20, 12), f"{dataset.name} - box AP @ IoU {iou_threshold:g} (same-class probe boxes merged with WBF)",
              fill="black", font=title)
    cols = [*dataset.labels, "mean"]
    name_w = 300
    col_w = min(220, (page_w - 40 - name_w) // len(cols))
    y = 12 + round(fs * 1.4) + line
    draw.text((20, y), "model", fill="#555555", font=font)
    for k, c in enumerate(cols):
        right = 20 + name_w + col_w * (k + 1)
        draw.text((right - draw.textlength(c, font=font), y), c, fill="#555555", font=font)
    y += line
    draw.line([20, y - 6, 20 + name_w + col_w * len(cols), y - 6], fill="#999999", width=1)
    for model, per_dataset in ap.items():
        per_class = per_dataset[dataset.name]
        if model == highlight:
            draw.rectangle([14, y - 4, 26 + name_w + col_w * len(cols), y + line - 8], fill="#EEF4FA")
        draw.text((20, y), model, fill="black", font=font)
        for k, v in enumerate([per_class[label] for label in dataset.labels] + [mean_ap(per_class)]):
            text = _fmt(v)
            right = 20 + name_w + col_w * (k + 1)
            draw.text((right - draw.textlength(text, font=font), y), text, fill="black", font=font)
        y += line
    note = ("AP: all merged boxes of a class ranked by score; a box is a hit if it overlaps an unmatched "
            "ground-truth box of the class by IoU >= the threshold; all-point interpolated precision-recall area.")
    draw.text((20, page_h - line), note, fill="#555555", font=ImageFont.load_default(round(fs * 0.8)))
    return page


def render_classification_page(
    cls: ClsTable, dataset: Dataset, highlight: str | None, cfg: dict[str, Any]
) -> Image.Image:
    """Return a page with one dataset's per-image classification stats: a row per model, a column per class.

    Each cell shows F1 with precision, recall and the threshold T; the
    `highlight` model's row is shaded.
    """
    page_w, page_h = cfg["page_size"]
    fs = cfg["cover_font_size"]
    line = round(fs * 2.6)
    font = ImageFont.load_default(fs)
    small = ImageFont.load_default(round(fs * 0.8))
    title = ImageFont.load_default(round(fs * 1.4))
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    draw.text((20, 12), f"{dataset.name} - per-image classification (best merged box > T; T maximises F1 per class)",
              fill="black", font=title)
    cols = [*dataset.labels, "mean F1"]
    name_w = 260
    col_w = min(330, (page_w - 40 - name_w) // len(cols))
    y = 12 + round(fs * 1.4) + round(fs * 1.6)
    draw.text((20, y), "model", fill="#555555", font=font)
    for k, c in enumerate(cols):
        draw.text((20 + name_w + col_w * k, y), c, fill="#555555", font=font)
    y += round(fs * 1.6)
    draw.line([20, y - 6, 20 + name_w + col_w * len(cols), y - 6], fill="#999999", width=1)
    for model, per_dataset in cls.items():
        per_class = per_dataset[dataset.name]
        if model == highlight:
            draw.rectangle([14, y - 4, 26 + name_w + col_w * len(cols), y + line - 10], fill="#EEF4FA")
        draw.text((20, y), model, fill="black", font=font)
        for k, label in enumerate(dataset.labels):
            c = per_class[label]
            x = 20 + name_w + col_w * k
            if c is None:
                draw.text((x, y), "-", fill="black", font=font)
                continue
            draw.text((x, y), f"F1 {c.f1:.3f}  AUROC {_fmt(c.auroc)}", fill="black", font=font)
            threshold = "-" if c.threshold is None else f"{c.threshold:.3f}"
            draw.text((x, y + round(fs * 1.15)),
                      f"base F1 {c.baseline_f1:.2f}  P {c.precision:.2f}  R {c.recall:.2f}  T {threshold}",
                      fill="#555555", font=small)
        draw.text((20 + name_w + col_w * len(dataset.labels), y), _fmt(mean_f1(per_class)), fill="black", font=font)
        y += line
    note = ("An image is predicted positive for a class if its best WBF-merged box of that class scores above T "
            "(0 if it has none); T is chosen per class to maximise F1. base F1 = every image labelled positive; "
            "AUROC uses the image scores directly.")
    draw.text((20, page_h - round(fs * 1.6)), note, fill="#555555", font=small)
    return page


def _label_colours(labels: list[str]) -> dict[str, str]:
    """Assign each label a palette colour, cycling if there are more labels than colours."""
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}


def _tagged_box(draw: ImageDraw.ImageDraw, box: list[float], text: str, colour: str, width: int,
                font: ImageFont.ImageFont) -> None:
    """Draw a box with a filled text tag above it (or inside, when it touches the top edge)."""
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline=colour, width=width)
    tag = draw.textbbox((0, 0), text, font=font)
    tag_h = tag[3] - tag[1] + 4
    ty = y0 - tag_h if y0 >= tag_h else y0
    draw.rectangle([x0, ty, x0 + tag[2] + 6, ty + tag_h], fill=colour)
    draw.text((x0 + 3, ty + 1), text, fill="black", font=font)


def render_page(
    image_path: Path, record: dict[str, Any], colours: dict[str, str], cfg: dict[str, Any],
    predictions: dict[str, list[Any]] | None = None, model: str | None = None,
) -> Image.Image:
    """Return one PDF page: the image with boxes drawn, under a header naming its labels.

    Without `predictions`, ground-truth boxes are drawn with label tags.
    With them, ground truth is drawn thin and untagged, and each class's top
    `draw_top_k` merged predictions are drawn thick with "label score" tags.
    Every page has the same fixed size, `cfg["page_size"]` (width, height in
    px); PDF viewers usually zoom to the page width, so a wide, short page
    takes up a predictable share of the screen. The image is scaled to fit
    below the header and centred horizontally.
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
        if predictions is None:
            _tagged_box(draw, [x, y, x + w, y + h], box["label"], colour, cfg["line_width"], font)
        else:
            draw.rectangle([x, y, x + w, y + h], outline=colour, width=max(1, cfg["line_width"] // 2))
    for label, boxes in (predictions or {}).items():
        for b in boxes[:cfg["draw_top_k"]]:
            _tagged_box(draw, [v * scale for v in (b.x0, b.y0, b.x1, b.y1)], f"{label} {b.score:.2f}",
                        colours[label], cfg["line_width"], font)

    page = Image.new("RGB", (page_w, page_h), "white")
    page.paste(img, ((page_w - img.width) // 2, header_h))
    hdraw = ImageDraw.Draw(page)
    labels = ", ".join(record["labels"]) or "(no labels)"
    hdraw.text((10, cfg["font_size"] // 2), f"{record['file']}    labels: {labels}", fill="black", font=font)
    second = (f"{model}: top {cfg['draw_top_k']} merged predictions per class (thick, scored); ground truth thin"
              if model else f"source: {record['source_id']}")
    hdraw.text((10, cfg["font_size"] * 3 // 2), second, fill="#555555", font=font)
    return page


def _save_pdf(pages: list[Image.Image], out: Path, cfg: dict[str, Any]) -> Path:
    """Save `pages` as a multi-page PDF at `out`."""
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=cfg["dpi"])
    return out


def render_reports(
    dataset_dir: Path, dataset: Dataset, all_stats: list[DatasetStats], ap: APTable, cls: ClsTable,
    predictions: dict[str, dict[str, Predictions]], iou_threshold: float, out_dir: Path, cfg: dict[str, Any],
) -> list[Path]:
    """Write a dataset's ground-truth report and one prediction report per evaluated model."""
    meta = yaml.safe_load((dataset_dir / "images.yaml").read_text())
    colours = _label_colours(meta["labels"])
    written = []
    pages = [render_cover(all_stats, dataset_dir.name, cfg)]
    if ap:
        pages.append(render_ap_page(ap, dataset, iou_threshold, None, cfg))
        pages.append(render_classification_page(cls, dataset, None, cfg))
    pages += [render_page(dataset_dir / r["file"], r, colours, cfg) for r in meta["images"]]
    written.append(_save_pdf(pages, out_dir / f"{dataset_dir.name}{cfg['pdf_suffix']}.pdf", cfg))
    for model, per_dataset in predictions.items():
        preds = per_dataset[dataset_dir.name]
        pages = [render_ap_page(ap, dataset, iou_threshold, model, cfg),
                 render_classification_page(cls, dataset, model, cfg)]
        pages += [render_page(dataset_dir / r["file"], r, colours, cfg, preds[str(dataset_dir / r["file"])], model)
                  for r in meta["images"]]
        written.append(_save_pdf(pages, out_dir / f"{dataset_dir.name}_{model}.pdf", cfg))
    return written


# ----- entry point ------------------------------------------------------------------


def clear_results(out_dir: Path, dataset_names: list[str], cfg: dict[str, Any]) -> list[Path]:
    """Delete the outputs of any previous run, so a (possibly partial) run reports only its own results.

    Only files this program writes are removed: `evaluation.yaml`, each
    dataset's ground-truth PDF and each dataset x model PDF, for every known
    dataset and model. The embedding cache and other files are kept.
    """
    candidates = [out_dir / "evaluation.yaml"]
    for name in dataset_names:
        candidates.append(out_dir / f"{name}{cfg['pdf_suffix']}.pdf")
        candidates += [out_dir / f"{name}_{model}.pdf" for model in MODELS]
    removed = [path for path in candidates if path.exists()]
    for path in removed:
        path.unlink()
    return removed



def main() -> None:
    """Evaluate each configured model on each sample dataset, print the AP table and write reports."""
    parser = argparse.ArgumentParser(description="Evaluate image probers' boxes on the sample datasets.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    parser.add_argument("--models", nargs="+", default=None, help=f"models to evaluate (default: test.models); "
                        f"choices: {', '.join(MODELS)}")
    parser.add_argument("--datasets", nargs="+", default=None, help="dataset names (default: all)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    cfg = config["test"]
    out_dir = Path(config["general"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir = Path(config["sample_data"]["datasets_dir"])
    dataset_dirs = sorted(p for p in datasets_dir.iterdir() if (p / "images.yaml").exists())
    all_stats = [dataset_stats(d) for d in dataset_dirs]
    removed = clear_results(out_dir, [d.name for d in dataset_dirs], cfg)
    print(f"cleared {len(removed)} result files from previous runs in {out_dir}")
    if args.datasets:
        dataset_dirs = [d for d in dataset_dirs if d.name in args.datasets]
    datasets = [Evaluate.load_dataset(d) for d in dataset_dirs]

    evaluator = Evaluate(WBF(**config["wbf"]), **config["evaluate"])
    ap: APTable = {}
    cls: ClsTable = {}
    predictions: dict[str, dict[str, Predictions]] = {}
    for model in args.models or cfg["models"]:
        prober = load_prober(model, config, cfg.get("box_threshold"))
        ap[model], cls[model], predictions[model] = {}, {}, {}
        for ds in datasets:
            preds = evaluator.predict(prober, ds.image_paths, ds.probe_paths, ds.probe_labels)
            predictions[model][ds.name] = preds
            ap[model][ds.name] = evaluator.average_precision(preds, ds.ground_truth, ds.labels)
            cls[model][ds.name] = evaluator.classification(preds, ds.ground_truth, ds.labels)
            print(f"[{model}] evaluated {ds.name}")
        # Free the model before loading the next one.
        del prober
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "iou_threshold": evaluator.iou_threshold,
        "box_threshold": cfg.get("box_threshold"),
        "wbf": config["wbf"],
        "ap": {m: {d: {"mean": mean_ap(per_class), "per_class": per_class} for d, per_class in per_ds.items()}
               for m, per_ds in ap.items()},
        "classification": {
            m: {d: {"mean_f1": mean_f1(per_class),
                    "per_class": {label: (None if c is None else asdict(c)) for label, c in per_class.items()}}
                for d, per_class in per_ds.items()}
            for m, per_ds in cls.items()},
        "predictions": {m: {d: predictions_to_dict(p) for d, p in per_ds.items()} for m, per_ds in predictions.items()},
    }
    out = out_dir / "evaluation.yaml"
    out.write_text(yaml.safe_dump(report, sort_keys=False, default_flow_style=None, width=140))
    print(f"\nwrote {out}")
    for dataset_dir, ds in zip(dataset_dirs, datasets):
        for path in render_reports(dataset_dir, ds, all_stats, ap, cls, predictions, evaluator.iou_threshold, out_dir, cfg):
            print(f"wrote {path}")
    # Results come last, after all progress output.
    print_tables(ap, datasets, evaluator.iou_threshold)
    print_classification_tables(cls, datasets)


if __name__ == "__main__":
    main()
