"""DINOv2 patch-feature matching as an `ImageProber`.

DINOv2 (Oquab et al., 2023) is the open-weight predecessor of DINOv3. The
matching method is identical (see `image_prober_dinov3`), so this class only
changes the default checkpoint and reads its own config section, which keeps
DINOv2's calibration separate from DINOv3's.
"""

from typing import Any

from .image_prober import run_cli
from .image_prober_dinov3 import ImageProber_DINOv3


class ImageProber_DINOv2(ImageProber_DINOv3):
    """Probe images by cosine similarity between a probe prototype and DINOv2 patch features."""

    config_section = "image_prober_dinov2"

    def __init__(
        self,
        repo_id: str = "facebook/dinov2-base",
        sim_offset: float = 0.25,
        sim_temperature: float = 0.05,
        **settings: Any,
    ) -> None:
        """Load DINOv2 (ungated) with its own calibration defaults; other `settings` as for `ImageProber_DINOv3`."""
        super().__init__(repo_id=repo_id, sim_offset=sim_offset, sim_temperature=sim_temperature, **settings)


if __name__ == "__main__":
    run_cli(ImageProber_DINOv2)
