import os
import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
from renderer.tensorboard import TensorBoard
from renderer.model import FCN
from renderer.stroke import draw

LOGS_DIR = os.path.join(os.path.dirname(__file__), ".logs/renderer/train")
PKL_PATH = os.path.join(os.path.dirname(__file__), ".pkls/renderer/renderer.pkl")
BATCH_SIZE = 64


def save_model(net: FCN, use_cuda: bool) -> None:
    if use_cuda:
        net.cpu()
    torch.save(net.state_dict(), PKL_PATH)
    if use_cuda:
        net.cuda()


def load_weights(net: FCN) -> None:
    pretrained_dict = torch.load(PKL_PATH)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


def train() -> None:
    # Create writer
    writer = TensorBoard(LOGS_DIR)

    # Initialize training
    criterion = nn.MSELoss()
    net = FCN()
    optimizer = optim.Adam(net.parameters(), lr=3e-6)

    # Determine if CUDA is available
    use_cuda = torch.cuda.is_available()
    print("Using CUDA:", use_cuda)

    # Try to load weights
    try:
        load_weights(net)
    except:
        print("No pretrained model found")

    # Train
    step = 0
    while step < 500000:
        net.train()
        train_batch = []
        ground_truth = []

        # Generate training set for batch
        for i in range(BATCH_SIZE):
            f = np.random.uniform(0, 1, 7)
            train_batch.append(f)
            ground_truth.append(draw(f))

        train_batch = torch.tensor(train_batch).float()
        ground_truth = torch.tensor(ground_truth).float()

        if use_cuda:
            net = net.cuda()
            train_batch = train_batch.cuda()
            ground_truth = ground_truth.cuda()

        # Generate predictions
        gen = net(train_batch)

        # # Calculate loss
        optimizer.zero_grad()
        loss = criterion(gen, ground_truth)
        loss.backward()
        optimizer.step()
        print("Step loss", step, loss.item())

        if step < 200000:
            lr = 1e-4
        elif step < 400000:
            lr = 1e-5
        else:
            lr = 1e-6
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ?Why are we saving train loss every step?
        writer.add_scalar("train/loss", loss.item(), step)

        # ?Why are we inferencing every 100?
        if step % 100 == 0:
            net.eval()
            gen = net(train_batch)
            loss = criterion(gen, ground_truth)
            writer.add_scalar("val/loss", loss.item(), step)
            for i in range(int(BATCH_SIZE / 2)):
                G = gen[i].cpu().data.numpy()
                GT = ground_truth[i].cpu().data.numpy()
                writer.add_image("train/gen{}.png".format(i), G, step)
                writer.add_image("train/ground_truth{}.png".format(i), GT, step)

        # Save every 1000
        if step % 1000 == 0:
            save_model(net, use_cuda)

        # Increment step
        step += 1
