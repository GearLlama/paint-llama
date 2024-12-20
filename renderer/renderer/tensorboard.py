from io import BytesIO

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np


class TensorBoard(object):
    def __init__(self, model_dir: str) -> None:
        self.summary_writer = SummaryWriter(model_dir)

    def add_image(self, tag: str, img: np.typing.NDArray, step: int) -> None:
        # summary = Summary()
        # bio = BytesIO()

        img = (img * 255).astype(np.uint8).reshape(3, 128, 128)

        # if type(img) == str:
        #     print("A")
        #     img = Image.open(img)
        # elif type(img) == Image.Image:
        #     print("B")
        #     pass
        # elif type(img) == np.ndarray:
        #     print("C")
        #     img = (img * 255).astype(np.uint8)
        # else:
        #     raise NotImplementedError(f"Image type not supported: {type(img)}")

        # img.save(bio, format="png")

        self.summary_writer.add_image(tag, img, step)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.summary_writer.add_scalar(tag, value, step)
