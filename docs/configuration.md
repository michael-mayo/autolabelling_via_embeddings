# Configuration

All key parameters live in `config.yaml`, with one section per module. Each
sample dataset's own build settings live in its `dataset.yaml`
([sample_datasets.md](sample_datasets.md)). The autolabeller's options live in
`autolabeller.yaml` ([autolabeller.md](autolabeller.md)).

| Section | Key | Description |
|---|---|---|
| `general` | `output_dir` | Directory for generated output (`scratch`) |
| `general` | `weights_dir` | Directory for downloaded weights and data (`weights`) |
| `general` | `random_seed` | Seed for reproducible runs |
| `sample_data` | `datasets_dir` | Folder of sample datasets (each sub-folder with a `dataset.yaml`) |
| `wbf` | `iou_threshold` | Overlap above which a box joins an existing fused box |
| `wbf` | `skip_box_threshold` | Input boxes scoring below this are dropped |
| `wbf` | `conf_type` | `avg` (mean score × min(boxes, sources) / sources) or `max` |
| `evaluate` | `iou_threshold` | IoU for a merged box to match a ground-truth box (AP@0.25) |
| `evaluate` | `label_method`, `label_top_k` | How a class's probe probabilities become an image confidence: `max`, `mean` or `topk_mean` (mean of the best k; default k = 2) |
| `test` | `models` | Models to evaluate, by registry name |
| `test` | `draw_top_k` | Merged boxes drawn per class per image in the model reports |
| `test` | `box_threshold` | If set, replaces every model's `box_threshold` during evaluation (`null` keeps each model's own) |
| `test` | `pdf_suffix` | Suffix for ground-truth PDF names (`<name><suffix>.pdf`) |
| `test` | `page_size` | Fixed PDF page size (px); 32:9 fills about half a 16:9 screen at fit-to-width |
| `test` | `line_width`, `font_size`, `cover_font_size`, `dpi` | PDF drawing settings |
| `image_prober` | `device` | `auto` (CUDA if available), `cuda` or `cpu` |
| `image_prober` | `batch_size` | Images or probes per encoder pass |
| `image_prober` | `memory_cache_items` | Embeddings kept in memory (LRU) |
| `image_prober` | `cache_dir` | On-disk embedding cache; `null` disables it |
| `image_prober_eval` | `box_iou` | Per-probe CLI: IoU for a returned box to count as a ground-truth hit |
| `image_prober_owlv2` | `repo_id` | Hugging Face model (base or large OWLv2) |
| `image_prober_owlv2` | `logit_offset`, `logit_temperature` | Calibration: `sigmoid((logit - offset) / temperature)` |
| `image_prober_owlv2` | `query_iou_fraction` | Probe query: highest-objectness box among those near the best overlap with the probe |
| `image_prober_sam3` | `exemplar_mode` | `canvas` (probes tiled under each image; default) or `transfer` (~chance) |
| `image_prober_sam3` | `strip_columns`, `max_strip_fraction` | Canvas: tiles per strip row; max share of canvas height for the strip |
| `image_prober_sam3` | `probe_batch_size` | Probes per DETR encoder/decoder pass |
| `image_prober_sam3` | `probability` | Transfer mode only: `max_score` or `presence` |
| `image_prober_dinov3`, `image_prober_dinov2` | `repo_id` | DINO checkpoint (DINOv3 gated; DINOv2 open) |
| `image_prober_dinov3`, `image_prober_dinov2` | `image_size`, `probe_size` | Longer side of images / probes fed to the backbone |
| `image_prober_dinov3`, `image_prober_dinov2` | `probe_border`, `top_k` | Probe margin trimmed before averaging; presence = mean of the best k patch similarities |
| `image_prober_dinov3`, `image_prober_dinov2` | `sim_offset`, `sim_temperature` | Calibration: `sigmoid((similarity - offset) / temperature)` |
| `image_prober_yoloe` | `weights_file` | Ultralytics YOLOE asset with a prompt-able NMS head (e.g. `yoloe-11l-seg.pt`) |
| `image_prober_yoloe` | `imgsz`, `probe_size` | Letterbox size; longer side of a probe on its canvas |
| `image_prober_embeddinggemma2` | `embedding_dim` | Matryoshka size: 768, 512, 256 or 128 |
| `image_prober_embeddinggemma2` | `tile_scales`, `tile_overlap`, `tile_batch_size` | Tile sides as fractions of the shorter image side; overlap; tiles per pass |
| `image_prober_embeddinggemma2` | `sim_offset`, `sim_temperature` | Calibration: `sigmoid((similarity - offset) / temperature)` |
| all model sections | `dtype`, `box_threshold`, `nms_iou`, `max_boxes` | GPU precision; box score threshold (0.5 for DINO, 0.01 otherwise); suppression overlap; cap |
