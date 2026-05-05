# SHOT 调制方式识别无源自适应

本项目实现调制方式识别任务中的无源自适应训练逻辑，主要参考论文：

```text
Do We Really Need to Access the Source Data?
Source Hypothesis Transfer for Unsupervised Domain Adaptation
```

当前实现重点是 SHOT 的 closed-set 训练流程：

1. 使用源域 `AWGN.dat` 训练源模型。
2. 丢弃源域数据，只保留源模型。
3. 在目标域 `Rayleigh*.dat` / `Rician*.dat` 上做无监督适配。
4. 目标域适配时冻结源分类器，只更新特征提取器。

## 项目结构

```text
SHOT/
  scripts/
    train_source.py      # 源域训练入口
    adapt_target.py      # 目标域 SHOT 适配入口
  shot/
    data.py              # .dat 数据读取和 6:2:2 划分
    models.py            # GNET 网络结构、SHOTNet、ADDAModel
    losses.py            # SHOT 损失
    pseudo_label.py      # 目标域原型伪标签
    train.py             # 源域训练和目标域适配循环
    evaluation.py        # SNR 准确率、t-SNE 特征收集和绘图
    results.py           # history.csv / metrics.json / 结果目录保存
  requirements.txt
  agentread.md
```

## 数据集

数据集位于项目上一级目录：

```text
../Datasets/
  AWGN.dat        # 源域
  Rayleigh1.dat   # 目标域
  Rayleigh2.dat   # 目标域
  Rayleigh3.dat   # 目标域
  Rician1.dat     # 目标域
  Rician3.dat     # 目标域
```

`.dat` 文件格式是 pickle 字典：

```text
(modulation_name, snr) -> ndarray[num_samples, 2, sample_len]
```

当前数据确认如下：

```text
类别数: 11
SNR 数: 20
每个 (mod, snr): 1000 条
输入形状: [2, 128]
完整单域样本数: 11 * 20 * 1000 = 220000
```

## 数据划分

按照每个 `(mod, snr)` 内部划分，保证每个类别和每个 SNR 都均衡：

```text
前 60%  -> train
中间 20% -> val
最后 20% -> test
```

完整单域默认数量：

```text
train: 132000
val:    44000
test:   44000
```

默认参数：

```text
--train-ratio 0.6
--val-ratio 0.2
```

## 网络结构

当前使用 1D GNET 风格网络：

```text
Input: [batch, 2, sample_len]
  -> GNETFeatureExtractor
       Conv1d + BatchNorm1d + LeakyReLU
       Conv1d + BatchNorm1d + LeakyReLU
       LSTM(sample_len -> 128, num_layers=2)
  -> flattened feature, dim = 32 * 128 = 4096
  -> GNETClassifierHead
       Linear 4096 -> 2048
       Linear 2048 -> 1024
       Linear 1024 -> 256
       Linear 256 -> num_classes
```

在 SHOT 中：

```text
GNETFeatureExtractor = 特征提取器 g
GNETClassifierHead   = 源假设分类器 h
```

源域训练阶段：

```text
feature_extractor: trainable
classifier:        trainable
```

目标域适配阶段：

```text
feature_extractor: trainable
classifier:        frozen
```

## 源域训练

使用 `AWGN.dat` 训练源模型：

```bash
python scripts/train_source.py \
  --data-root ../Datasets/AWGN.dat \
  --output checkpoints/source.pt
```

限制 SNR 或调制类别：

```bash
python scripts/train_source.py \
  --data-root ../Datasets/AWGN.dat \
  --snrs=-6,-4,-2,0,2,4,6,8,10,12 \
  --mods BPSK,QPSK,8PSK,QAM16,QAM64 \
  --output checkpoints/source.pt
```

源域训练会保存：

```text
checkpoints/source.pt
results/source/history.csv
results/source/metrics.json
```

## 如何开启训练

第一次运行前先安装依赖：

```bash
pip install -r requirements.txt
```

然后训练源域模型：

```bash
python scripts/train_source.py \
  --data-root ../Datasets/AWGN.dat \
  --output checkpoints/source.pt \
  --epochs 20 \
  --batch-size 32
```

源模型训练完成后，对单个目标域做无源自适应：

```bash
python scripts/adapt_target.py \
  --data-root ../Datasets/Rayleigh1.dat \
  --source-checkpoint checkpoints/source.pt \
  --target-split train \
  --eval-split val \
  --epochs 15 \
  --batch-size 32
```

如果想使用全部目标域无标签数据做适配：

```bash
python scripts/adapt_target.py \
  --data-root ../Datasets/Rayleigh1.dat \
  --source-checkpoint checkpoints/source.pt \
  --target-split all \
  --eval-split val \
  --epochs 15 \
  --batch-size 32
```

依次适配所有目标域：

```bash
python scripts/adapt_target.py --data-root ../Datasets/Rayleigh1.dat --source-checkpoint checkpoints/source.pt
python scripts/adapt_target.py --data-root ../Datasets/Rayleigh2.dat --source-checkpoint checkpoints/source.pt
python scripts/adapt_target.py --data-root ../Datasets/Rayleigh3.dat --source-checkpoint checkpoints/source.pt
python scripts/adapt_target.py --data-root ../Datasets/Rician1.dat --source-checkpoint checkpoints/source.pt
python scripts/adapt_target.py --data-root ../Datasets/Rician3.dat --source-checkpoint checkpoints/source.pt
```

训练完成后查看结果：

```text
results/source/
results/AWGN_to_Rayleigh1/
results/AWGN_to_Rayleigh2/
results/AWGN_to_Rayleigh3/
results/AWGN_to_Rician1/
results/AWGN_to_Rician3/
```

### Windows PowerShell 版本

在 Windows PowerShell 中进入项目目录：

```powershell
cd "D:\研究内容2无源自适应\SHOT"
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

训练源域模型：

```powershell
python .\scripts\train_source.py `
  --data-root ..\Datasets\AWGN.dat `
  --output .\checkpoints\source.pt `
  --epochs 20 `
  --batch-size 32
```

适配单个目标域：

```powershell
python .\scripts\adapt_target.py `
  --data-root ..\Datasets\Rayleigh1.dat `
  --source-checkpoint .\checkpoints\source.pt `
  --target-split train `
  --eval-split val `
  --epochs 15 `
  --batch-size 32
```

依次适配所有目标域，可以直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1
```

如果你的 Python 命令不是 `python`，例如使用 Conda 环境里的解释器：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1 `
  -Python "D:\miniconda3\envs\shot\python.exe"
```

缩短测试运行时间：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1 `
  -SourceEpochs 1 `
  -TargetEpochs 1 `
  -BatchSize 16
```

## 目标域适配

以 `Rayleigh1.dat` 为例：

```bash
python scripts/adapt_target.py \
  --data-root ../Datasets/Rayleigh1.dat \
  --source-checkpoint checkpoints/source.pt
```

默认设置：

```text
--target-split all
--eval-split val
```

也可以只在目标域 train 划分上适配，并在 val 上评估：

```bash
python scripts/adapt_target.py \
  --data-root ../Datasets/Rayleigh1.dat \
  --source-checkpoint checkpoints/source.pt \
  --target-split train \
  --eval-split val
```

目标域标签只用于评估、每个 SNR 准确率和 t-SNE，不参与训练损失。

## SHOT 目标损失

目标域适配时使用：

```text
L = L_information_maximization + beta * L_pseudo_label_ce
```

其中：

```text
L_information_maximization = conditional entropy + diversity
L_pseudo_label_ce          = 基于目标域类别原型伪标签的交叉熵
```

关键约束：

```text
目标域适配不能访问源域数据
目标域适配不能更新 classifier
目标域标签不能参与 loss
```

## 结果保存

每个目标域单独保存一个结果文件夹：

```text
results/
  AWGN_to_Rayleigh1/
    target_checkpoint.pt
    history.csv
    metrics.json
    accuracy_by_snr.csv
    tsne_before.png
    tsne_after.png
```

文件说明：

```text
target_checkpoint.pt   # 适配后的目标模型
history.csv            # 每轮 loss / IM loss / 伪标签 CE / eval acc
metrics.json           # 配置、类别、划分比例、历史指标
accuracy_by_snr.csv    # 每个 SNR 下的目标域准确率
tsne_before.png        # 域适应前目标特征 t-SNE
tsne_after.png         # 域适应后目标特征 t-SNE
```

如果只想保存数值结果，不画 t-SNE：

```bash
python scripts/adapt_target.py \
  --data-root ../Datasets/Rayleigh1.dat \
  --source-checkpoint checkpoints/source.pt \
  --no-tsne
```

如果不想使用目标标签做任何评估：

```bash
python scripts/adapt_target.py \
  --data-root ../Datasets/Rayleigh1.dat \
  --source-checkpoint checkpoints/source.pt \
  --no-eval-target-labels
```

## 依赖

```bash
pip install -r requirements.txt
```

当前环境需要安装 PyTorch 后才能实际训练。
