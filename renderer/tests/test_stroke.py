import numpy as np
from renderer.stroke import draw


def test_draw_succeeds() -> None:
    f = np.random.uniform(0, 1, 7)
    canvas = draw(f)
