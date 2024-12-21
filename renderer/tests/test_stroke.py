import numpy as np
from renderer.stroke import draw


def test_draw_succeeds() -> None:
    f = np.random.uniform(0, 1, 6)
    img = draw(f)
    assert img.shape == (128, 128)
