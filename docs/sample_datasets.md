# Sample datasets

Three small datasets from different domains, used to evaluate automatic
labelling. Each has **20 probes** (labelled object crops, split evenly across
its classes) and **80 images** (unlabelled full images, with hidden labels
kept for evaluation only).

| Name | Source | Labels | License |
|---|---|---|---|
| `coco_objects` | COCO val2017 (`rafaelpadilla/coco2017`) | dog, cat, bicycle, bus, umbrella | CC BY 4.0 |
| `road_cracks` | RDD2022 (`dronefreak/RDD2022`) | longitudinal_crack, transverse_crack, alligator_crack | CC BY-SA 4.0 |
| `dental_xray` | DENTEX (`liodon-ai/dental-panoramic-xray-yolo`) | caries, periapical_lesion, impacted_tooth | CC BY 4.0 |

Ground-truth summary:

| Dataset | Boxes | Boxes per image | Images with no labels | Multi-label images |
|---|---|---|---|---|
| `coco_objects` | 174 | 2.2 | 8 | 23 |
| `road_cracks` | 181 | 2.3 | 8 | 41 |
| `dental_xray` | 419 | 5.2 | 0 | 43 |

Per-class counts are on the cover page of each `scratch/<dataset>_ground_truth.pdf`.

## How the datasets are built

`python -m src.sample_data` builds every folder under `sample_datasets/` that
contains a `dataset.yaml`. It downloads the source dataset from Hugging Face
(cached in `weights/sample_data_cache/`), then:

- **Probes:** one box per probe, balanced across labels, each from a
  different source image and no smaller than `probe_min_side` pixels. Each is
  cropped with 10% context.
- **Images:** drawn round-robin over label combinations, so single- and
  multi-label images both appear and no class dominates. About 10% are
  negatives (none of the labels) where the source has them.
- **No overlap:** probes are cropped from source images that are never used
  as dataset images.
- **One scale:** every image is resized to 1280 px wide, keeping its aspect
  ratio. Probes are cropped from source images resized the same way.

Each run replaces the generated files (`probes/`, `images/`, `probes.yaml`,
`images.yaml`) and leaves `dataset.yaml` untouched.

Notes on the sources:

- **RDD2022 "pothole":** dropped. In this mirror that class mostly marks
  blurred road markings, crosswalks and manholes.
- **BCCD blood cells:** rejected as the medical dataset, because every image
  contains all three cell types.

## Folder layout

```
sample_datasets/<name>/
  dataset.yaml        # hand-written build settings: source, labels, counts, image width, seed
  probes.yaml         # labels list; per probe: file, label, source_id, source_bbox_xywh
  probes/probeN.jpg   # labelled object crops
  images/imageN.jpg   # full, unlabelled images
  images.yaml         # hidden per-image labels and boxes, for evaluation only
```

Each dataset folder is self-contained. To add one, create a folder holding a
`dataset.yaml` and run the builder. To remove one, delete its folder.

## `dataset.yaml` keys

| Key | Description |
|---|---|
| `source` | Loader: `hf_rows`, `rdd2022` or `yolo_hub`, plus its own keys (`dataset`, `split`, ...) |
| `labels` | Source category names to keep; `label_map` optionally renames them |
| `license` | License recorded in the generated YAML files |
| `num_probes` | Probe crops, split evenly across labels |
| `num_images` | Full images |
| `image_width` | Width (px) every image is resized to, keeping aspect ratio |
| `negative_fraction` | Share of images with none of the labels |
| `probe_padding` | Context around probe boxes (fraction of box size) |
| `probe_min_side` | Min shorter side of a probe box, in source-image pixels |
| `jpeg_quality` | JPEG quality for images and probe crops |
| `seed` | Random seed for reproducible selection |
