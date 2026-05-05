from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    num_classes: int
    sample_len: int
    backbone: str = "cnn"


class GNETFeatureExtractor(nn.Module):
    def __init__(self, sample_len: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )

        self.conv = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )

        self.rnn = nn.LSTM(sample_len, 128, num_layers=2, batch_first=True)
        self.out_dim = 32 * self.rnn.hidden_size

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.conv(x)
        # Keep channel dimension as sequence length, matching the original GNET layout.
        x, _ = self.rnn(x)
        return x.contiguous().view(x.size(0), -1)


class GNETClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.Dropout(0.6),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.6),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.6),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes),
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_feature_extractor(sample_len: int, backbone: str) -> nn.Module:
    if backbone == "cnn":
        return GNETFeatureExtractor(sample_len)

    raise ValueError("backbone must be 'cnn'")


class SHOTNet(nn.Module):
    """SHOT model using the GNET feature extractor and classifier hypothesis."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.feature_extractor = build_feature_extractor(config.sample_len, config.backbone)
        self.classifier = GNETClassifierHead(
            self.feature_extractor.out_dim,
            config.num_classes,
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        logits = self.classifier(features)
        return features, logits

    def freeze_classifier(self) -> None:
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False


class ADDAModel(nn.Module):
    """
    ADDA model with the same GNET modules.

    Network structure:
        - source_encoder and target_encoder use the GNET encoder stack.
        - classifier uses the GNET classifier head.
        - discriminator is the ADDA domain head.

    The difference is the training procedure:
        - source_encoder + classifier are trained first.
        - target_encoder is initialized from source_encoder.
        - source_encoder and classifier are frozen.
        - target_encoder is adversarially trained against discriminator.
    """

    def __init__(self, num_classes: int, sample_len: int, backbone: str = "cnn"):
        super().__init__()

        self.source_encoder = build_feature_extractor(sample_len, backbone)
        self.target_encoder = build_feature_extractor(sample_len, backbone)

        dim = self.source_encoder.out_dim
        self.classifier = GNETClassifierHead(dim, num_classes)

        self.discriminator = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def copy_source_to_target(self) -> None:
        self.target_encoder.load_state_dict(
            copy.deepcopy(self.source_encoder.state_dict())
        )

    def source_logits(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.source_encoder(x)
        return self.classifier(feat)

    def target_logits(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.target_encoder(x)
        return self.classifier(feat)


def build_model(
    num_classes: int,
    sample_len: int,
    backbone: str = "cnn",
) -> SHOTNet:
    return SHOTNet(
        ModelConfig(
            num_classes=num_classes,
            sample_len=sample_len,
            backbone=backbone,
        )
    )
