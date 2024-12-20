import torch
import numpy as np

import torch.nn as nn
from renderer.tensorboard import TensorBoard
from renderer.model import FCN
from renderer.stroke import draw

writer = TensorBoard("./logs/train_log/")
import torch.optim as optim

criterion = nn.MSELoss()
net = FCN()
optimizer = optim.Adam(net.parameters(), lr=3e-6)
batch_size = 16

use_cuda = torch.cuda.is_available()
print("Using CUDA:", use_cuda)
step = 0


def save_model():
    if use_cuda:
        net.cpu()
    torch.save(net.state_dict(), "./pkls/renderer.pkl")
    if use_cuda:
        net.cuda()


def load_weights():
    pretrained_dict = torch.load("./pkls/renderer.pkl")
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


try:
    load_weights()
except:
    print("No pretrained model found")


while step < 500000:
    net.train()
    train_batch = []
    ground_truth = []
    for i in range(batch_size):
        f = np.random.uniform(0, 1, 7)
        train_batch.append(f)
        ground_truth.append(draw(f))

    train_batch = torch.tensor(train_batch).float()
    ground_truth = torch.tensor(ground_truth).float()

    if use_cuda:
        net = net.cuda()
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
    gen = net(train_batch)
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
    writer.add_scalar("train/loss", loss.item(), step)
    if step % 100 == 0:
        net.eval()
        gen = net(train_batch)
        loss = criterion(gen, ground_truth)
        writer.add_scalar("val/loss", loss.item(), step)
        for i in range(int(batch_size / 2)):
            G = gen[i].cpu().data.numpy()
            GT = ground_truth[i].cpu().data.numpy()
            writer.add_image("train/gen{}.png".format(i), G, step)
            writer.add_image("train/ground_truth{}.png".format(i), GT, step)
    if step % 1000 == 0:
        save_model()
    step += 1