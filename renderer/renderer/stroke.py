from typing import Tuple
from math import floor
import cv2
import numpy as np


def normal(x: float, width: int):
    return (int)(x * (width - 1) + 0.5)


def draw(
    command: np.typing.NDArray[np.float64],
    brush_radius: float = 0.05,
    width: int = 128,
):
    # Decompose command tuple
    x0, y0, x1, y1, x2, y2 = command

    # Set opacity and brush radius
    z0 = z2 = brush_radius

    # Normalize coordinates
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)

    # Create canvas
    canvas = np.zeros([width * 2, width * 2, 3]).astype("float32")

    # Create stroke
    tmp = 1.0 / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
        y = (int)((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
        z = (int)((1 - t) * z0 + t * z2)
        cv2.circle(canvas, center=(y, x), radius=z, color=color, thickness=-1)  # type: ignore

    bitmap = cv2.resize(canvas, dsize=(width, width))
    print(command, bitmap.shape)
    return 1 - bitmap


if __name__ == "__main__":
    img = 1 - draw(np.random.uniform(0, 1, 7))
    cv2.imwrite("test.png", img)
