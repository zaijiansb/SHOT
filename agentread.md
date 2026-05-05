1.除了代码外，使用中文；
2.我要完成是的是调制方式识别的无源自适应；
3.让你读的论文，都是学习他们的方法，
4.代码写的要有模块化，将模型和训练、损失，画图、保存等分别放在不同的文件；
5.网络结构从下面中选取对应的模块：

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

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_feature_extractor(sample_len: int, backbone: str) -> nn.Module:
    if backbone == "cnn":
        return GNETFeatureExtractor(sample_len)

    raise ValueError("backbone must be 'cnn'")


class ADDAModel(nn.Module):
    """
    ADDA model.

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



6.数据集
Datasets中包含的文件名称：
   源域 ：AWGN.dat
   目标域：Rayleigh1.dat
    Rayleigh2.dat
    Rayleigh3.dat
    Rician1.dat
    Rician3.dat


7.数据的划分，按照每个 (mod, snr) 内部划分，来作为训练集、验证集和测试集（6：2：2）


8.结果保存时，每个目标域划分一个文件夹，来保存其文件和结果

9.保存每个 SNR 下的准确率，以及域适应前后的T-SNE


10.给readme中加入开启训练的代码，linux版本和windows版本都需要