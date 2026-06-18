# 进食声音识别项目

本项目是一个基于 PyTorch 的进食声音分类系统。项目将原始音频转换为 Mel 频谱图，再使用 CNN 模型完成 20 类进食/食物相关声音识别，并提供训练、预测、实验指标生成和 Word 报告自动生成脚本。

## 项目功能

- 支持 20 类音频分类：芦荟、汉堡、卷心菜、蜜饯、胡萝卜、薯片、巧克力、饮品、薯条、葡萄、软糖、冰淇淋、果冻、面条、腌菜、披萨、排骨、三文鱼、汤类、鸡翅。
- 使用 Mel 频谱图作为音频特征。
- 支持 CNN V1 和 CNN V2 两种模型结构，默认使用 `cnn_v2`。
- 支持 CUDA GPU 加速、AMP 混合精度训练、类别权重、标签平滑、早停和学习率调度。
- 支持 Mel 频谱缓存，第一次处理数据后，后续训练速度更快。
- 支持单条音频 Top-5 预测输出。
- 支持生成实验图表、评估指标和课程报告素材。

## 项目结构

```text
.
├── config.py                  # 项目配置：数据路径、类别、音频参数、训练参数
├── dataset.py                 # 音频数据读取、Mel 频谱提取、缓存、数据增强、DataLoader
├── model.py                   # CNN V1 / CNN V2 模型结构
├── train.py                   # 模型训练入口
├── predict.py                 # 单条音频预测入口
├── generate_report_assets.py  # 生成实验指标、图表和报告素材
├── build_report_docx.py       # 生成 Word 项目报告
├── report_assets/             # 实验指标和图表素材
└── 报告资产/                  # 报告相关资料
```

## 环境依赖

建议使用 Python 3.9 或以上版本，并安装以下依赖：

```bash
pip install torch torchaudio matplotlib numpy scikit-learn python-docx
```

如果需要使用 GPU，请根据自己的 CUDA 版本安装对应的 PyTorch 和 torchaudio。

## 数据集目录要求

`config.py` 中的 `DATA_DIR` 需要指向原始音频数据集目录。该目录下应按类别名建立子文件夹，例如：

```text
clips_rd/
├── aloe/
├── burger/
├── cabbage/
├── candied_fruits/
├── carrots/
├── chips/
├── chocolate/
├── drinks/
├── fries/
├── grapes/
├── gummies/
├── ice-cream/
├── jelly/
├── noodles/
├── pickles/
├── pizza/
├── ribs/
├── salmon/
├── soup/
└── wings/
```

每个类别文件夹中放入对应的 `.wav`、`.mp3`、`.flac` 或 `.ogg` 音频文件。

## 修改配置

训练前请先打开 `config.py`，确认以下路径符合本机实际情况：

```python
DATA_DIR = r"D:\昆明城市学院\音频识别\archive\clips_rd"
MODEL_SAVE_PATH = r"D:\昆明城市学院\音频识别\best_model.pth"
MEL_CACHE_DIR = r"D:\昆明城市学院\音频识别\mel_cache"
```

如果换到其他电脑运行，需要把这三个路径改成本机存在的路径。

## 训练模型

在项目根目录运行：

```bash
python train.py
```

训练流程包括：

1. 扫描音频数据集。
2. 自动生成 Mel 频谱缓存。
3. 按类别分层划分训练集和验证集。
4. 创建 CNN 模型。
5. 训练并保存验证集准确率最高的模型。

训练完成后，最佳模型会保存到 `MODEL_SAVE_PATH` 指定的位置。

## 使用模型预测

训练完成并生成 `best_model.pth` 后，可以运行：

```bash
python predict.py D:/test/chips_sample.wav
```

程序会输出 Top-5 预测类别和置信度，例如：

```text
排名 类别                置信度
1    chips              98.23%
2    fries               0.91%
3    wings               0.32%
```

## 生成实验素材

运行：

```bash
python generate_report_assets.py
```

该脚本会加载最佳模型，并在验证集上生成：

- 总体准确率
- 每类 precision、recall、f1-score
- 混淆矩阵
- 类别分布图
- 训练曲线图
- `report_assets/metrics.json`

## 生成 Word 报告

运行：

```bash
python build_report_docx.py
```

脚本会读取报告模板和 `report_assets` 中的实验素材，生成项目报告：

```text
进食声音识别项目报告.docx
```

## 当前实验结果

根据仓库中的 `report_assets/metrics.json`，当前最佳模型信息如下：

- 模型结构：`cnn_v2`
- 最佳 epoch：76
- 验证集样本数：2236
- 训练集样本数：8904
- 总样本数：11140
- 验证准确率：96.74%
- 加权 F1-score：96.73%

## 注意事项

- 仓库中的路径目前使用的是 Windows 绝对路径，换电脑运行前必须修改 `config.py`。
- 训练依赖原始音频数据集，单独下载代码无法直接训练。
- 第一次运行训练时会生成 Mel 频谱缓存，耗时会比较长。
- 如果 Windows 下 DataLoader 报多进程错误，可以把 `config.py` 中的 `NUM_WORKERS` 改为 `0`。
- 如果没有 GPU，也可以使用 CPU 运行，但训练速度会明显变慢。

## 后续优化方向

- 将绝对路径改为相对路径，提高项目可移植性。
- 增加 `requirements.txt`，方便一键安装依赖。
- 增加命令行参数，例如自定义数据集路径、训练轮数和模型保存路径。
- 增加测试集划分，避免只用验证集报告最终效果。
- 增加 Gradio 或 PyQt 图形界面，便于用户上传音频并查看预测结果。
