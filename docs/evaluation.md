# Evaluation

`src/evaluate.py` (class `Evaluate`) evaluates any prober through the public
`ImageProber` interface, and `src/test.py` runs it for every model and
dataset. Evaluation keeps two outputs strictly apart:

| Output | Built from | Merging | Metric |
|---|---|---|---|
| **Image labels** | each probe's `probability` | `label_confidences`: combine a class's probe probabilities | baseline F1, tuned F1, AUROC per class |
| **Box predictions** | each probe's `boxes` | `merge_boxes`: weighted boxes fusion (WBF) | box AP at IoU 0.25 per class |

`Evaluate.run(prober, images, probes, probe_labels)` probes once and returns
both outputs.

## Image labelling

### Combining a class's probes

An image probed with the N probes of a class, P0 … PN-1, gets N
probabilities. `Evaluate.label_confidences` combines them into one confidence:

- `max`: the highest probability. One matching probe is enough, which suits
  classes that look varied. But it is fragile to one bad probe, and on images
  without the class it rises as N grows.
- `mean`: the average. It rewards agreement and is robust to an outlier
  probe, but it dilutes a genuine match by a single probe.
- `topk_mean`: the mean of the best k. It needs some agreement without
  requiring all probes to agree.

Experiment (mean AUROC over all 6 models and 3 datasets):

| Method | Overall | coco_objects | dental_xray | road_cracks | Best in how many of 18 model-dataset pairs |
|---|---|---|---|---|---|
| **topk_mean, k = 2** (default) | **0.687** | 0.929 | 0.534 | 0.596 | 5 |
| max | 0.683 | 0.914 | 0.537 | 0.598 | 5 |
| mean | 0.677 | 0.939 | 0.514 | 0.578 | 8 |

The differences are about 0.01, so the methods are effectively tied. Mean is
best on COCO; max is best on cracks and X-rays. `topk_mean` with k = 2 is the
best compromise and is the default (`evaluate.label_method`, `evaluate.label_top_k`).

### Metrics

An image is truly positive for a class if it has a ground-truth box of that
class. It is predicted positive if its confidence exceeds a threshold T. Per
class, the following are reported:

- **Baseline F1**: the F1 from labelling every image positive, 2p / (1 + p)
  for prevalence p. Recall is 1 and precision is p, so this is the score to
  beat. It is high when most images contain the class (caries is in 55 of 80
  dental images, giving 0.815).
- **Tuned F1**: the F1 at the T that maximises it, with that T and its
  precision and recall. T is chosen from 0 and the distinct confidences.
  Because T is fitted on the evaluated images, this is optimistic.
- **AUROC**: threshold-free; the chance that a random positive image
  outscores a random negative one, with ties counted half.

## Box predictions

### Weighted boxes fusion (`src/wbf.py`)

`WBF` (Solovyev et al., 2021) merges boxes from several sources. For us, each
probe of a class is one source, including probes that returned no boxes.

1. Boxes are visited from highest to lowest score. Each joins the existing
   cluster it overlaps most by IoU > `iou_threshold` (0.55), or starts a new
   cluster.
2. A cluster's box is the score-weighted mean of its members' coordinates.
3. With `conf_type: avg`, its score is the mean member score × min(members, N) / N.
   This scores down boxes that only some probes found. For example, with
   4 probes, one box at 0.9 scores 0.225, while four overlapping boxes at
   0.6–0.9 score 0.75.

`WBF` works on plain `(x0, y0, x1, y1, score)` tuples and doesn't depend on
the rest of the project. `Evaluate.merge_boxes` is the only box-merging step,
applied the same way to every model.

### Box AP

All merged boxes of a class, across all images, are ranked by score. Each is
a hit if it overlaps a not-yet-matched ground-truth box of that class by IoU
>= 0.25 (`evaluate.iou_threshold`). AP is the area under the precision-recall
curve with all-point interpolation, as in PASCAL VOC 2010+.

Each model's `box_threshold` controls which candidate boxes reach WBF:

- **Detectors** (OWLv2, SAM 3, YOLOE, EmbeddingGemma 2) use 0.01, so AP sees
  low-scoring boxes too.
- **DINO** uses 0.5, because its threshold decides which patches join a region.

Mean box AP at IoU 0.25:

| Model | coco_objects | dental_xray | road_cracks |
|---|---|---|---|
| SAM 3 | **0.566** | **0.060** | 0.013 |
| DINOv2 | 0.450 | 0.000 | 0.020 |
| RAD-DINO | 0.017 | 0.003 | 0.016 |
| DINOv3 | 0.438 | 0.000 | 0.031 |
| OWLv2 | 0.307 | 0.001 | 0.026 |
| YOLOE | 0.303 | 0.000 | 0.009 |
| EmbeddingGemma 2 | 0.173 | 0.039 | **0.050** |

Box quality depends on where the boxes come from. OWLv2, SAM 3 and YOLOE
predict real object boxes. DINO boxes are limited to the patch grid (about
20 px). EmbeddingGemma 2 boxes are fixed tiles.

## `python -m src.test` outputs

Each run first deletes the previous run's outputs, so a partial run
(`--models`, `--datasets`) reports only its own results. The embedding cache
is kept.

- **Console**, printed at the end of the run:
  - a box-AP table per dataset (models × classes) and a mean box-AP summary;
  - a classification table per dataset (model × class rows: images, baseline
    F1, tuned F1, AUROC, T, precision, recall);
  - a summary of means over classes under an every-image-positive baseline
    row.
- **`scratch/evaluation.yaml`:** the settings (IoU, WBF, label method), all
  tables and every model's merged box predictions.
- **`scratch/<dataset>_ground_truth.pdf`:** a cover summarising all datasets,
  a box-AP page, a classification page, then every image with its ground truth.
- **`scratch/<dataset>_<model>.pdf`:** the box-AP and classification pages with
  the model's row shaded, then every image with the ground truth (thin) and the
  model's top `test.draw_top_k` merged boxes per class (thick, with scores).

A full run takes about two minutes with a warm embedding cache. The 21 PDFs
total about 140 MB.

## Findings and next steps

- **Everyday photos work well.** DINOv2 reaches image-label AUROC 0.976 and
  tuned F1 0.903 on COCO.
- **Cracks and X-rays are weak.** Road cracks reach at best AUROC 0.64.
  Dental X-rays are near chance (0.43–0.63), and tuned F1 barely moves above
  the every-image-positive baseline.
- **The features do carry location signal.** With DINO, patches inside
  ground-truth boxes score higher than patches outside. The problem is
  discrimination: every X-ray is full of teeth and every road is asphalt.
- **A radiograph backbone alone doesn't help.** RAD-DINO (DINOv2 trained on
  chest X-rays) reaches dental AUROC 0.528, below DINOv3 (0.564) and
  EmbeddingGemma 2 (0.632). Its patch features are very similar everywhere:
  the median presence similarity is 0.618 for present pairs and 0.612 for
  absent ones. On COCO it is near chance (0.544), as expected.
- **Most promising next steps:**
  - a per-dataset patch classifier on DINOv3 features, trained against the
    other classes' probes as negatives;
  - higher input resolution for small and thin targets;
  - a dental-specific backbone, if one with public weights appears.
