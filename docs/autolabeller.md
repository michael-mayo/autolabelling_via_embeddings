# Autolabeller

`python -m src.autolabeller` labels a folder of unlabelled images from a
folder of labelled probe crops, and writes the labels in a format Label
Studio can import. All options are in `autolabeller.yaml`. By default it
labels the `coco_objects` sample images from that dataset's probes using
DINOv2.

## How it works

1. **Probes:** labels are read from `probes_yaml` (the sample datasets'
   `probes.yaml` format). If `probes_yaml` is `null`, each probe's label is the
   name of its sub-folder: `probes_dir/<label>/*.jpg`.
2. **Probing:** one `image_probe` call scores every probe against every image
   with the chosen model (`model`, with its settings in `model_config`). The
   embedding cache is shared with `src.test`.
3. **Confidences:** each class's probe probabilities are combined into one
   confidence per image (`label_method`: `max`, `mean` or `topk_mean`).
4. **Labels:** an image gets every class whose confidence exceeds
   `threshold`, or its entry in `class_thresholds`.
5. **Output:** the labels are written in Label Studio's import format. A
   matching labelling interface, `label_config.xml`, is written next to them.

## Output formats

The format follows the extension of `output`.

**`.json` (default, `scratch/labels.json`)** gives one Label Studio task per
image. The labels are a pre-annotation, so they show up already selected,
ready to accept or correct:

```json
{
  "data": {
    "image": "/data/local-files/?d=sample_datasets/coco_objects/images/image10.jpg",
    "confidence_dog": 0.1135, "confidence_cat": 0.0984, "confidence_bicycle": 0.9847,
    "confidence_bus": 0.9998, "confidence_umbrella": 0.4871
  },
  "predictions": [{
    "model_version": "dinov2-topk_mean2-t0.5",
    "score": 0.9923,
    "result": [{"from_name": "label", "to_name": "image", "type": "choices",
                "value": {"choices": ["bicycle", "bus"]}}]
  }]
}
```

- **Prediction `score`:** the mean confidence of the chosen labels, or the
  highest confidence if no label was chosen. In Label Studio's data manager
  you can sort by it, or by any `confidence_<class>` column, to review the
  least certain images first.
- **No labels:** an image with no labels gets an empty prediction.

**`.csv`** has one row per image, with the columns `image`, `labels`
(comma-separated) and `confidence_<class>`. Label Studio imports CSV columns
as task data only. It can't import pre-annotations from CSV, so the labels
appear as a text column rather than as selected choices.

## Importing into Label Studio

1. **Serve the images.** Start Label Studio with local-file serving rooted at
   the project folder, which matches the default `url_prefix` and `url_root`:
   ```bash
   LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
   LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd) label-studio start
   ```
   In the project's settings, add a local-files storage under *Cloud
   Storage* whose path lies under that root (for example
   `<project>/sample_datasets/coco_objects/images`), but don't sync it.
   Importing `labels.json` creates the tasks.
2. **Set the labelling interface.** Under *Settings → Labeling Interface →
   Code*, paste the contents of `scratch/label_config.xml`. Its control
   names must match `image_key` and `choices_name`.
3. **Import the labels.** Import `scratch/labels.json`.

If the images are served from somewhere else, such as a web server or a
bucket, set `url_prefix` to its base URL and `url_root` to the local folder
that maps to that URL.

## Options (`autolabeller.yaml`)

| Key | Default | Description |
|---|---|---|
| `model` | `dinov2` | Prober, by registry name: `owlv2`, `sam3`, `dinov3`, `dinov2`, `raddino`, `yoloe`, `embeddinggemma2` |
| `model_config` | `config.yaml` | Model settings: the `image_prober` and `image_prober_<model>` sections |
| `probes_dir` | `sample_datasets/coco_objects/probes` | Folder of labelled probe crops |
| `probes_yaml` | `sample_datasets/coco_objects/probes.yaml` | Probe labels as `probes: [{file, label}]`. Each `file` is looked up under `probes_dir`, then relative to the YAML file. `null` takes labels from sub-folder names |
| `images_dir` | `sample_datasets/coco_objects/images` | Folder of images to label |
| `recursive` | `false` | Also label images in sub-folders |
| `extensions` | `.jpg` … `.webp` | Image file extensions to include |
| `label_method`, `label_top_k` | `topk_mean`, `2` | How a class's probe probabilities become one confidence (see [evaluation.md](evaluation.md)) |
| `threshold` | `0.5` | Confidence above which an image gets a label |
| `class_thresholds` | `{}` | Per-class thresholds that override `threshold` |
| `output` | `scratch/labels.json` | Output file; `.json` or `.csv` |
| `label_config` | `label_config.xml` | Labelling-interface file, written next to `output`; `null` skips it |
| `image_key`, `choices_name` | `image`, `label` | Label Studio names for the image data key and the Choices control |
| `url_prefix`, `url_root` | `/data/local-files/?d=`, `.` | Image URL = `url_prefix` + the image path relative to `url_root` |

## Choosing thresholds

With the defaults on `coco_objects` (DINOv2, `topk_mean`, threshold 0.5),
label precision is 0.62, recall is 0.94, and 33 of 80 images get exactly the
right labels. A single threshold of 0.5 errs towards labelling too much.
`python -m src.test` reports each class's F1-maximising threshold (column T)
on the sample datasets. On your own data, label a few dozen images by hand
and use them to set `class_thresholds`.
