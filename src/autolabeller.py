"""Automatic multi-label image labelling from labelled probe crops, exported for Label Studio.

`AutoLabeller` reads its options from `autolabeller.yaml`. It probes every
unlabelled image with every labelled probe using one of the `ImageProber`
models, combines each class's probe probabilities into one confidence per
image (`evaluate.label_confidences`: max, mean or mean of the top k), and
gives each image every class whose confidence exceeds its threshold.

The labels are written in a Label Studio import format chosen by the output
file's extension:

- `.json`: a list of tasks, each with the image URL and per-class
  confidences as task data and the labels as a pre-annotation (a
  `choices` prediction), ready to review and accept;
- `.csv`: one row per image (image URL, labels, per-class confidences).
  Label Studio imports CSV rows as task data only, without pre-annotations.

A matching labelling interface (`label_config.xml`) is written next to the
output, for pasting into the Label Studio project's settings.

Run with `python -m src.autolabeller [--config autolabeller.yaml]`.
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any
from xml.sax.saxutils import quoteattr

import yaml

from .evaluate import label_confidences
from .image_prober import ImageProber
from .image_prober import load_prober

# Default options file, relative to the working directory.
DEFAULT_CONFIG = "autolabeller.yaml"
# Output formats, by file extension.
OUTPUT_FORMATS = (".json", ".csv")
# Decimal places kept for confidences in the output files.
CONFIDENCE_DECIMALS = 4


class AutoLabeller:
    """Label a directory of images from a directory of labelled probes, for import into Label Studio."""

    def __init__(self, options: dict[str, Any], prober: ImageProber | None = None) -> None:
        """Take the parsed `autolabeller.yaml` options; build the prober from them unless one is given.

        The prober is built from `options["model"]` and the settings in
        `options["model_config"]` (a `config.yaml`). Passing `prober` skips
        that, e.g. to reuse a loaded model across calls.
        """
        self.options = options
        suffix = Path(options["output"]).suffix.lower()
        if suffix not in OUTPUT_FORMATS:
            raise ValueError(f"output must end in one of {OUTPUT_FORMATS}, not {options['output']!r}")
        if prober is None:
            model_config = yaml.safe_load(Path(options["model_config"]).read_text())
            prober = load_prober(options["model"], model_config)
        self.prober = prober

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG) -> "AutoLabeller":
        """Build an autolabeller from an options file."""
        return cls(yaml.safe_load(Path(path).read_text()))

    # ----- inputs ---------------------------------------------------------------------

    def load_probes(self) -> dict[str, str]:
        """Return {probe path: label} for every probe.

        Labels come from `probes_yaml` when set, otherwise from each probe's
        sub-folder name under `probes_dir`.
        """
        probes_dir = Path(self.options["probes_dir"])
        if self.options.get("probes_yaml"):
            yaml_path = Path(self.options["probes_yaml"])
            entries = yaml.safe_load(yaml_path.read_text())["probes"]
            probes = {}
            for entry in entries:
                # A file is looked up under probes_dir, then relative to the YAML file.
                candidates = [probes_dir / entry["file"], probes_dir / Path(entry["file"]).name,
                              yaml_path.parent / entry["file"]]
                found = next((c for c in candidates if c.is_file()), None)
                if found is None:
                    raise FileNotFoundError(f"probe {entry['file']!r} not found under {probes_dir} "
                                            f"or {yaml_path.parent}")
                probes[str(found)] = str(entry["label"])
        else:
            probes = {str(path): path.parent.name for path in self._image_files(probes_dir, recursive=True)
                      if path.parent != probes_dir}
        if not probes:
            raise ValueError(f"no probes found in {probes_dir}")
        return probes

    def find_images(self) -> list[str]:
        """Return the paths of the images to label, sorted."""
        images = [str(p) for p in self._image_files(Path(self.options["images_dir"]),
                                                     self.options.get("recursive", False))]
        if not images:
            raise ValueError(f"no images found in {self.options['images_dir']}")
        return images

    def _image_files(self, folder: Path, recursive: bool) -> list[Path]:
        """Image files in `folder` (and its sub-folders if `recursive`) with an allowed extension."""
        extensions = {e.lower() for e in self.options["extensions"]}
        paths = folder.rglob("*") if recursive else folder.iterdir()
        return sorted((p for p in paths if p.is_file() and p.suffix.lower() in extensions), key=_natural_key)

    # ----- labelling ------------------------------------------------------------------

    def confidences(self, images: list[str], probes: dict[str, str]) -> dict[str, dict[str, float]]:
        """Return {image path: {label: confidence}} from one `image_probe` call over all images and probes."""
        results = self.prober.image_probe(images, list(probes))
        return {str(image): label_confidences(per_probe, probes, self.options["label_method"],
                                              self.options["label_top_k"])
                for image, per_probe in results.items()}

    def labels(self, confidences: dict[str, float]) -> list[str]:
        """The labels whose confidence exceeds their threshold (`class_thresholds`, else `threshold`)."""
        thresholds = self.options.get("class_thresholds") or {}
        return [label for label, conf in confidences.items()
                if conf > thresholds.get(label, self.options["threshold"])]

    def image_url(self, image: str) -> str:
        """The URL Label Studio loads an image from: `url_prefix` + its path relative to `url_root`.

        An image outside `url_root` keeps its absolute path.
        """
        path = Path(image).resolve()
        root = Path(self.options["url_root"]).resolve()
        relative = path.relative_to(root) if path.is_relative_to(root) else path
        return f"{self.options['url_prefix']}{relative.as_posix()}"

    # ----- outputs --------------------------------------------------------------------

    def tasks(self, confidences: dict[str, dict[str, float]], class_names: list[str]) -> list[dict[str, Any]]:
        """Label Studio tasks: image URL and confidences as data, labels as a `choices` prediction.

        The prediction's score is the mean confidence of the chosen labels
        (or the highest confidence when none is chosen), so tasks can be
        sorted by it in Label Studio for review.
        """
        opts = self.options
        tasks = []
        for image, per_class in confidences.items():
            chosen = self.labels(per_class)
            score = (sum(per_class[c] for c in chosen) / len(chosen)) if chosen else max(per_class.values())
            result = [{
                "from_name": opts["choices_name"], "to_name": opts["image_key"], "type": "choices",
                "value": {"choices": chosen},
            }] if chosen else []
            data = {opts["image_key"]: self.image_url(image)}
            data |= {f"confidence_{c}": round(per_class[c], CONFIDENCE_DECIMALS) for c in class_names}
            tasks.append({"data": data, "predictions": [{
                "model_version": self.model_version, "score": round(score, CONFIDENCE_DECIMALS),
                "result": result,
            }]})
        return tasks

    @property
    def model_version(self) -> str:
        """Prediction model version recorded in Label Studio, e.g. `dinov2-topk_mean2-t0.5`."""
        opts = self.options
        method = opts["label_method"] + (str(opts["label_top_k"]) if opts["label_method"] == "topk_mean" else "")
        return f"{opts['model']}-{method}-t{opts['threshold']}"

    def write_json(self, path: Path, confidences: dict[str, dict[str, float]], class_names: list[str]) -> None:
        """Write Label Studio JSON tasks with pre-annotations."""
        path.write_text(json.dumps(self.tasks(confidences, class_names), indent=2))

    def write_csv(self, path: Path, confidences: dict[str, dict[str, float]], class_names: list[str]) -> None:
        """Write one CSV row per image: image URL, comma-separated labels, per-class confidences."""
        image_key = self.options["image_key"]
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([image_key, "labels"] + [f"confidence_{c}" for c in class_names])
            for image, per_class in confidences.items():
                writer.writerow([self.image_url(image), ",".join(self.labels(per_class))]
                                + [round(per_class[c], CONFIDENCE_DECIMALS) for c in class_names])

    def write_label_config(self, path: Path, class_names: list[str]) -> None:
        """Write a Label Studio labelling interface: the image with one multiple-choice control."""
        opts = self.options
        choices = "\n".join(f"    <Choice value={quoteattr(c)}/>" for c in class_names)
        path.write_text(
            "<View>\n"
            f"  <Image name={quoteattr(opts['image_key'])} value={quoteattr('$' + opts['image_key'])}/>\n"
            f"  <Choices name={quoteattr(opts['choices_name'])} toName={quoteattr(opts['image_key'])} "
            'choice="multiple" showInLine="true">\n'
            f"{choices}\n"
            "  </Choices>\n"
            "</View>\n")

    # ----- pipeline -------------------------------------------------------------------

    def run(self) -> Path:
        """Label every image and write the output (and labelling interface); return the output path."""
        start = time.perf_counter()
        probes = self.load_probes()
        images = self.find_images()
        # Classes in probe order, so the output columns follow probes_yaml.
        class_names = list(dict.fromkeys(probes.values()))
        print(f"{len(images)} images, {len(probes)} probes, {len(class_names)} classes "
              f"({', '.join(class_names)}); model {self.options['model']}")
        confidences = self.confidences(images, probes)

        output = Path(self.options["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".json":
            self.write_json(output, confidences, class_names)
        else:
            self.write_csv(output, confidences, class_names)
        if self.options.get("label_config"):
            self.write_label_config(output.parent / self.options["label_config"], class_names)

        counts = {c: sum(c in self.labels(per_class) for per_class in confidences.values()) for c in class_names}
        unlabelled = sum(not self.labels(per_class) for per_class in confidences.values())
        print("images per label: " + ", ".join(f"{c} {n}" for c, n in counts.items())
              + f"; no label: {unlabelled}")
        print(f"wrote {output} in {time.perf_counter() - start:.1f} s")
        return output


def _natural_key(path: Path) -> list[int | str]:
    """Sort key that orders numbers within names numerically (image2 before image10)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="options file (default: %(default)s)")
    args = parser.parse_args()
    AutoLabeller.from_yaml(args.config).run()


if __name__ == "__main__":
    main()
