import numpy as np
import torch
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()


def prRed(prt: str) -> None:
    print(f"\033[91m {prt}\033[00m")


def prGreen(prt: str) -> None:
    print(f"\033[92m {prt}\033[00m")


def prYellow(prt: str) -> None:
    print(f"\033[93m {prt}\033[00m")


def prLightPurple(prt: str) -> None:
    print(f"\033[94m {prt}\033[00m")


def prPurple(prt: str) -> None:
    print(f"\033[95m {prt}\033[00m")


def prCyan(prt: str) -> None:
    print(f"\033[96m {prt}\033[00m")


def prLightGray(prt: str) -> None:
    print(f"\033[97m {prt}\033[00m")


def prBlack(prt: str) -> None:
    print(f"\033[98m {prt}\033[00m")


def to_numpy(var: torch.Tensor) -> np.typing.NDArray:
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray: np.typing.NDArray, device: str) -> torch.Tensor:
    return torch.tensor(ndarray, dtype=torch.float, device=device)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()  # pylint: disable=protected-access
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
