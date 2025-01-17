# TODO: : In need of refactor - variables not propertly encapsulated
from typing import Optional, Tuple, Annotated
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from renderer.model import StrokeFCN
from renderer.train import PKL_PATH as RENDERER_PKL_PATH
from rl_painter.actor import ActorResNet
from rl_painter.critic import CriticResNet
from rl_painter.replay_memory import ReplayMemory
from rl_painter.wgan import cal_reward, load_gan, save_gan
from rl_painter.wgan import update_gan as _update_gan
from rl_painter.utils import DEVICE, hard_update, soft_update, to_tensor, to_numpy
from rl_painter.tensorboard import Writer


criterion = nn.MSELoss()

Decoder = StrokeFCN()
Decoder.load_state_dict(torch.load(RENDERER_PKL_PATH))


def decode(x: torch.Tensor, canvas: Annotated[torch.Tensor, torch.float32, 3, 128, 128]) -> torch.Tensor:
    # TODO: Modify color choice to be discrete
    x = x.view(-1, 6 + 3)  # 6 stroke parameters + 3 color parameters
    stroke = 1 - Decoder(x[:, :6])
    stroke = stroke.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
    return canvas


class AgentDDPG:

    def __init__(
        self,
        batch_size: int = 64,
        env_batch: int = 1,
        max_step: int = 40,
        tau: float = 0.001,
        discount: float = 0.9,
        rmsize: int = 800,
        writer: Optional[Writer] = None,
        resume: Optional[str] = None,
    ) -> None:
        # TODO: Should DDPG be saving the model??
        self.coord = self.get_default_coord(DEVICE)
        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size

        self.actor = ActorResNet(9, 18, 65)  # target, canvas, stepnum, coordconv 3 + 3 + 1 + 2
        self.actor_target = ActorResNet(9, 18, 65)
        self.critic = CriticResNet(3 + 9, 18, 1)  # add the last canvas for better prediction
        self.critic_target = CriticResNet(3 + 9, 18, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-2)

        if resume is not None:
            self.load_weights(resume)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = ReplayMemory(rmsize * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        if writer is None:
            raise ValueError("Writer is required")
        self.writer = writer
        self.log = 0

        self.state = [None] * self.env_batch  # Most recent state
        self.action = [None] * self.env_batch  # Most recent action
        self.noise_level = np.zeros(self.env_batch)  # TODO: What should it be initialized as?
        self.choose_device()

    @staticmethod
    def get_default_coord(device) -> torch.Tensor:
        # TODO: Add type hints
        coord = torch.zeros([1, 2, 128, 128])
        for ix in range(128):
            for iy in range(128):
                coord[0, 0, ix, iy] = ix / 127.0
                coord[0, 1, ix, iy] = iy / 127.0
        coord = coord.to(device)
        return coord

    def play(self, state: torch.Tensor, target=False) -> torch.Tensor:
        # Convert state from uint to float
        state = torch.cat(
            (
                state[:, :6].float() / 255,
                state[:, 6:7].float() / self.max_step,
                self.coord.expand(state.shape[0], 2, 128, 128),
            ),
            1,
        )
        if target:
            return self.actor_target(state)
        else:
            return self.actor(state)

    def update_gan(self, state: torch.Tensor) -> None:
        canvas = state[:, :3]
        gt = state[:, 3:6]
        fake, real, penal = _update_gan(canvas.float() / 255, gt.float() / 255)
        if self.log % 20 == 0:
            self.writer.add_scalar("train/gan_fake", fake, self.log)
            self.writer.add_scalar("train/gan_real", real, self.log)
            self.writer.add_scalar("train/gan_penal", penal, self.log)

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, target=False) -> Tuple[torch.Tensor, torch.Tensor]:
        T = state[:, 6:7]
        gt = state[:, 3:6].float() / 255
        canvas0 = state[:, :3].float() / 255
        canvas1 = decode(action, canvas0)
        gan_reward = cal_reward(canvas1, gt) - cal_reward(canvas0, gt)
        # L2_reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)
        coord_ = self.coord.expand(state.shape[0], 2, 128, 128)
        merged_state = torch.cat([canvas0, canvas1, gt, (T + 1).float() / self.max_step, coord_], 1)
        # canvas0 is not necessarily added
        if target:
            Q = self.critic_target(merged_state)
            return (Q + gan_reward), gan_reward
        else:
            Q = self.critic(merged_state)
            if self.log % 20 == 0:
                self.writer.add_scalar("train/expect_reward", Q.mean(), self.log)
                self.writer.add_scalar("train/gan_reward", gan_reward.mean(), self.log)
            return (Q + gan_reward), gan_reward

    def update_policy(self, lr: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.log += 1

        for param_group in self.critic_optim.param_groups:
            param_group["lr"] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group["lr"] = lr[1]

        # Sample batch
        state, action, _, next_state, terminal = self.memory.sample_batch(self.batch_size, DEVICE)

        self.update_gan(next_state)

        with torch.no_grad():
            next_action = self.play(next_state, True)
            target_q, _ = self.evaluate(next_state, next_action, True)
            target_q = self.discount * ((1 - terminal.float()).view(-1, 1)) * target_q

        cur_q, step_reward = self.evaluate(state, action)
        target_q += step_reward.detach()

        value_loss = criterion(cur_q, target_q)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        action = self.play(state)
        pre_q, _ = self.evaluate(state.detach(), action)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return -policy_loss, value_loss

    def observe(self, reward, state, done) -> None:
        # TODO: Add type hints - FYI, removed step because it wasn't used
        s0 = torch.tensor(self.state, device="cpu")
        a = to_tensor(self.action, "cpu")  # type: ignore
        r = to_tensor(reward, "cpu")
        s1 = torch.tensor(state, device="cpu")
        d = to_tensor(done.astype("float32"), "cpu")
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    def noise_action(self, action):
        # TODO: Add type hints
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype("float32")
        return np.clip(action.astype("float32"), 0, 1)

    def select_action(self, state: torch.Tensor, noise_factor: int = 0):
        # TODO: Add type hints
        # TODO: removed "return_fix" as it didn't make sense to me
        self.eval()
        with torch.no_grad():
            action = self.play(state)
            action = to_numpy(action)
        if noise_factor > 0:
            action = self.noise_action(action)
        self.train()
        self.action = action
        return self.action

    def reset(self, observation: torch.Tensor, factor: int) -> None:
        self.state = observation
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def load_weights(self, path):
        # TODO: Add type hints
        if path is None:
            return
        self.actor.load_state_dict(torch.load(f"{path}/actor.pkl"))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pkl"))
        load_gan(path)

    def save_model(self, path):
        # TODO: Add type hints
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(), f"{path}/actor.pkl")
        torch.save(self.critic.state_dict(), f"{path}/critic.pkl")
        save_gan(path)
        self.choose_device()

    def eval(self) -> None:
        # Switch models to evaluation mode
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self) -> None:
        # Swtich models to training mode
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def choose_device(self) -> None:
        Decoder.to(DEVICE)
        self.actor.to(DEVICE)
        self.actor_target.to(DEVICE)
        self.critic.to(DEVICE)
        self.critic_target.to(DEVICE)
