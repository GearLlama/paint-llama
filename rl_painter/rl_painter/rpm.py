from typing import List, Tuple, Union
import random
import torch


class ReplayMemory:
    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.buffer: List[torch.Tensor] = []
        self.index = 0

    def append(self, obj: torch.Tensor) -> None:
        if self.size() > self.buffer_size:
            print("buffer size larger than set value, trimming...")
            self.buffer = self.buffer[(self.size() - self.buffer_size) :]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)

    def size(self) -> int:
        return len(self.buffer)

    def sample_batch(
        self, batch_size: int, device: int, only_state=False
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        if only_state:
            res = torch.stack(tuple(item[3] for item in batch), dim=0)
            return res.to(device)
        else:
            res = []
            for i in range(5):
                k = torch.stack(tuple(item[i] for item in batch), dim=0)
                res.append(k.to(device))
            return res[0], res[1], res[2], res[3], res[4]
