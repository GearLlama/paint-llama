"""Tensorboard monitoring setup."""

import numpy as np
from tensorboardX import SummaryWriter


class Writer:
    def __init__(self, model_dir: str) -> None:
        self.summary_writer = SummaryWriter(model_dir)

    def add_image(self, tag: str, img: np.typing.NDArray, step: int) -> None:
        img = (img * 255).reshape(1, 128, 128).astype(np.uint8)
        self.summary_writer.add_image(tag, img, step)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.summary_writer.add_scalar(tag, value, step)
