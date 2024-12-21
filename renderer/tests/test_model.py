import numpy as np
import torch

from renderer.model import StrokeFCN


def test_draw_succeeds() -> None:
    # Init model
    network = StrokeFCN()

    # Create training data
    training_data = [np.random.uniform(0, 1, 6)]
    training_data = torch.tensor([np.random.uniform(0, 1, 6)]).float()

    # Train
    prediction = network(training_data)[0]

    # Assert shape of prediction is correct
    assert prediction.shape == (128, 128)
