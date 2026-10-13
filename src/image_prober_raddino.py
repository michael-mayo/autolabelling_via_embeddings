"""RAD-DINO patch-feature matching as an `ImageProber`.

RAD-DINO (Perez-Garcia et al., 2025; Microsoft Health Futures) is DINOv2-base
trained further with the DINOv2 objective on about 880k chest X-rays. It is a
radiograph-domain backbone: the closest openly available one to our dental
panoramic X-rays, though its training anatomy differs. The matching method is
identical to DINOv3's (see `image_prober_dinov3`); this subclass overrides
only what differs:

- the checkpoint and calibration defaults, read from its own config section;
- the downloaded files (the repo also holds a DINO training head and a
  duplicate backbone, which are not needed);
- the input normalisation: RAD-DINO's greyscale statistics, not ImageNet's.

The weights are MIT-licensed; the model card restricts use to research, not
clinical practice.
"""

from pathlib import Path
from typing import Any

import torch

from .image_prober import run_cli
from .image_prober_dinov3 import ImageProber_DINOv3

# Files needed to load the backbone with AutoModel.
WEIGHT_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]
# RAD-DINO's input normalisation (greyscale statistics of its training X-rays, repeated per channel).
RAD_DINO_MEAN = [0.5307, 0.5307, 0.5307]
RAD_DINO_STD = [0.2583, 0.2583, 0.2583]


class ImageProber_RADDINO(ImageProber_DINOv3):
    """Probe images by cosine similarity between a probe prototype and RAD-DINO patch features."""

    config_section = "image_prober_raddino"

    def __init__(
        self,
        repo_id: str = "microsoft/rad-dino",
        sim_offset: float = 0.615,
        sim_temperature: float = 0.01,
        **settings: Any,
    ) -> None:
        """Load RAD-DINO (ungated) and switch to its normalisation; other `settings` as for `ImageProber_DINOv3`."""
        super().__init__(repo_id=repo_id, sim_offset=sim_offset, sim_temperature=sim_temperature, **settings)
        self._mean = torch.tensor(RAD_DINO_MEAN, device=self.device).view(3, 1, 1)
        self._std = torch.tensor(RAD_DINO_STD, device=self.device).view(3, 1, 1)

    def ensure_weights(
        self, repo_id: str, allow_patterns: list[str] | None = None, ignore_patterns: list[str] | None = None
    ) -> Path:
        """Download only the backbone files (see `WEIGHT_FILES`), not the repo's other checkpoints."""
        return super().ensure_weights(repo_id, allow_patterns=WEIGHT_FILES, ignore_patterns=ignore_patterns)


if __name__ == "__main__":
    run_cli(ImageProber_RADDINO)
