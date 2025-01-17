from typing import Optional
import torch
import numpy as np
import cv2
from torchvision import transforms
from rl_painter.agent import decode
from rl_painter.utils import DEVICE, to_numpy
from rl_painter.tensorboard import Writer


# TODO make configurable with pydantic settings
WIDTH = 128
CANVAS_AREA = WIDTH * WIDTH
NEURAL_RENDERER_INPUT_SIZE = 6
MNT_BASE = "/mnt/f"
BASE_DIR = f"{MNT_BASE}/paint_llama/rl_painter"
CELEBA_DIR = f"{MNT_BASE}/img_align_celeba/img_align_celeba"

augment_image = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
    ]
)


class PaintEnvionment:
    def __init__(self, batch_size: int, max_step: int) -> None:
        self.batch_size = batch_size
        self.max_step = max_step
        self.action_space = NEURAL_RENDERER_INPUT_SIZE + 3  # TODO: What is this used for?
        self.observation_space = (self.batch_size, WIDTH, WIDTH, 7)  # 3 canvas + 3 img + 1 stepnum
        self.canvas = torch.zeros([batch_size, 3, WIDTH, WIDTH], dtype=torch.uint8).to(DEVICE)
        self.target_image = torch.zeros([self.batch_size, 3, WIDTH, WIDTH], dtype=torch.uint8).to(DEVICE)
        self.test = False
        self.train_num = 0
        self.test_num = 0
        self.img_train = []
        self.img_test = []
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
            ]
        )

    def load_data(self) -> None:
        # CelebA
        for i in range(3000):
            im_i = i + 1
            img_id = f"{im_i:06d}"
            try:
                img = cv2.imread(  # pylint: disable=no-member # type: ignore
                    f"{CELEBA_DIR}/{img_id}.jpg", cv2.IMREAD_UNCHANGED  # pylint: disable=no-member # type: ignore
                )
                img = cv2.resize(img, (WIDTH, WIDTH))  # pylint: disable=no-member # type: ignore
                if i > 2000:
                    self.train_num += 1
                    self.img_train.append(img)
                else:
                    self.test_num += 1
                    self.img_test.append(img)
            finally:
                if (im_i) % 10000 == 0:
                    print(f"loaded {im_i} images")
        print(f"finish loading data, {self.train_num} training images, {self.test_num} testing images")

    def pre_data(self, id: int, test: bool) -> np.ndarray:
        if test:
            img = self.img_test[id]
        else:
            img = self.img_train[id]
        if not test:
            img = augment_image(img)
        img = np.asarray(img)
        return np.transpose(img, (2, 0, 1))

    def reset(self, test: bool = False, begin_num: int = 0) -> torch.Tensor:
        # TODO: Are these type hints correct?
        self.test = test
        self.imgid = [0] * self.batch_size
        self.target_image = torch.zeros([self.batch_size, 3, WIDTH, WIDTH], dtype=torch.uint8).to(DEVICE)
        for i in range(self.batch_size):
            if test:
                id = (i + begin_num) % self.test_num
            else:
                id = np.random.randint(self.train_num)
            self.imgid[i] = id
            self.target_image[i] = torch.tensor(self.pre_data(id, test))
        self.tot_reward = ((self.target_image.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0
        self.canvas = torch.zeros([self.batch_size, 3, WIDTH, WIDTH], dtype=torch.uint8).to(DEVICE)
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()

    def observation(self) -> torch.Tensor:
        # canvas * 3 color channels * width * width
        # target image * 3 color channels * width * width
        # T (step num) * 1 * width * width
        T = torch.ones([self.batch_size, 1, WIDTH, WIDTH], dtype=torch.uint8) * self.stepnum
        return torch.cat((self.canvas, self.target_image, T.to(DEVICE)), 1)  # canvas, img, T

    def cal_trans(self, s, t):
        # TODO: Add type hints
        return (s.transpose(0, 3) * t).transpose(0, 3)

    def step(self, action):
        # TODO: Add type hints
        self.canvas = (decode(action, self.canvas.float() / 255) * 255).byte()
        self.stepnum += 1
        ob = self.observation()
        done = self.stepnum == self.max_step
        reward = self.cal_reward()  # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None

    def cal_dis(self):
        # TODO: Add type hints
        return (((self.canvas.float() - self.target_image.float()) / 255) ** 2).mean(1).mean(1).mean(1)

    def cal_reward(self):
        # TODO: Add type hints
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return to_numpy(reward)


class FastPaintEnvironment:
    def __init__(
        self,
        max_episode_length: int = 10,
        env_batch: int = 64,
        writer: Optional[Writer] = None,
    ) -> None:
        self.max_episode_length = max_episode_length
        self.env_batch = env_batch
        self.env = PaintEnvionment(self.env_batch, self.max_episode_length)
        self.env.load_data()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        if writer is None:
            raise ValueError("Writer is required")
        self.writer = writer
        self.test = False
        self.log = 0

    def save_image(self, log, step):
        # TODO: Add type hints
        for i in range(self.env_batch):
            if self.env.imgid[i] <= 10:
                canvas = cv2.cvtColor((to_numpy(self.env.canvas[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                self.writer.add_image("{}/canvas_{}.png".format(str(self.env.imgid[i]), str(step)), canvas, log)
        if step == self.max_episode_length:
            for i in range(self.env_batch):
                if self.env.imgid[i] < 50:
                    gt = cv2.cvtColor((to_numpy(self.env.gt[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                    canvas = cv2.cvtColor((to_numpy(self.env.canvas[i].permute(1, 2, 0))), cv2.COLOR_BGR2RGB)
                    self.writer.add_image(str(self.env.imgid[i]) + "/_target.png", gt, log)
                    self.writer.add_image(str(self.env.imgid[i]) + "/_canvas.png", canvas, log)

    def step(self, action):
        # TODO: Add type hints
        with torch.no_grad():
            ob, r, d, _ = self.env.step(torch.tensor(action).to(DEVICE))
        if d[0]:
            if not self.test:
                self.dist = self.get_dist()
                for i in range(self.env_batch):
                    self.writer.add_scalar("train/dist", self.dist[i], self.log)
                    self.log += 1
        return ob, r, d, _

    def get_dist(self):
        # TODO: Add type hints
        return to_numpy((((self.env.gt.float() - self.env.canvas.float()) / 255) ** 2).mean(1).mean(1).mean(1))

    def reset(self, test=False, episode=0) -> torch.Tensor:
        self.test = test
        ob = self.env.reset(self.test, episode * self.env_batch)
        return ob
