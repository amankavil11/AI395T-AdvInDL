from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.requires_grad_(False)

        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        torch.nn.init.kaiming_normal_(self.lora_a.weight)

        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)
        torch.nn.init.zeros_(self.lora_b.weight)

        self.lora_a.weight.requires_grad_(True)
        self.lora_b.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=x.device)
        out_base = super().forward(x)
        out_lora = self.lora_b(self.lora_a(x))
        return (out_base + out_lora).to(x.dtype)


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim=lora_dim, group_size=group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim=lora_dim, group_size=group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim=lora_dim, group_size=group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim=lora_dim, group_size=group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim=lora_dim, group_size=group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim=lora_dim, group_size=group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim=lora_dim, group_size=group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim=lora_dim, group_size=group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim=lora_dim, group_size=group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
