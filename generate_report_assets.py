# generate_report_assets.py
# ============================================================
# 报告素材生成脚本：评估模型、导出指标 JSON、绘制报告图片
# ============================================================
#
# 这个文件主要服务于课程报告，不直接参与训练。
# 它会读取训练日志、加载 best_model.pth、在验证集上重新评估，
# 然后生成类别分布图、训练曲线图、混淆矩阵图和模型结构示意图。

# 从 pathlib 导入 Path，用来处理项目路径和素材输出路径。
from pathlib import Path

# 导入 json，用来把评估指标保存成 metrics.json。
import json

# 导入 re，用正则表达式从训练日志文本中提取 epoch 指标。
import re

# 导入 sys，用来把项目根目录加入 Python 模块搜索路径。
import sys

# 导入 matplotlib 主包，先设置无界面绘图后再导入 pyplot。
import matplotlib

# 使用 Agg 后端，表示在没有图形界面的环境中也能保存图片。
matplotlib.use("Agg")

# 导入 pyplot，用来创建和保存图表。
import matplotlib.pyplot as plt

# 导入 numpy，用于生成坐标数组和处理矩阵。
import numpy as np

# 导入准确率、分类报告和混淆矩阵计算函数。
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 导入 PyTorch，用来加载模型 checkpoint 和执行验证集推理。
import torch

# 导入 DataLoader，用来批量遍历验证集。
from torch.utils.data import DataLoader


# BASE_DIR 是当前脚本所在目录，也就是项目根目录。
BASE_DIR = Path(__file__).resolve().parent

# ASSET_DIR 是报告图片和 metrics.json 的输出目录。
ASSET_DIR = BASE_DIR / "report_assets"

# 确保素材目录存在；exist_ok=True 表示已经存在也不报错。
ASSET_DIR.mkdir(exist_ok=True)

# 把项目根目录加入模块搜索路径，保证后面能 import config、dataset、model。
sys.path.insert(0, str(BASE_DIR))

# 导入配置模块；里面有路径、类别、音频参数和设备设置。
import config  # noqa: E402

# 从 dataset.py 导入缓存数据集、验证集包装器和分层划分函数。
from dataset import PrecomputedMelDataset, ValMelDataset, _stratified_split  # noqa: E402

# 从 model.py 导入模型构建函数。
from model import build_model  # noqa: E402


# 定义训练日志解析函数；它会扫描项目根目录下的 txt 文件。
def parse_training_logs():
    """从训练留痕 txt 中提取每轮 loss 和 accuracy，供训练曲线图使用。"""
    # 编译正则表达式，匹配 train.py 打印的 Epoch 指标行。
    pattern = re.compile(
        # 匹配 Epoch [001/080] 这类轮次信息。
        r"Epoch\s*\[(\d{1,3})/(\d{1,3})\].*?"

        # 匹配 train_loss、train_acc、val_loss、val_acc 四个数字。
        r"train_loss=([0-9.]+)\s+train_acc=([0-9.]+)%\s+"
        r"val_loss=([0-9.]+)\s+val_acc=([0-9.]+)%"
    )

    # runs 用来保存每个日志文件解析出的训练记录。
    runs = []

    # 遍历项目根目录下所有 .txt 文件。
    for path in sorted(BASE_DIR.glob("*.txt")):
        # 读取日志文本；errors="ignore" 可以跳过少量编码异常字符。
        text = path.read_text(encoding="utf-8", errors="ignore")

        # records 保存当前日志文件中的逐 epoch 指标。
        records = []

        # 在文本中查找所有匹配的 Epoch 行。
        for match in pattern.finditer(text):
            # 把正则匹配到的字符串转换成结构化字典。
            records.append(
                {
                    # 当前 epoch 编号。
                    "epoch": int(match.group(1)),

                    # 总 epoch 数。
                    "total_epochs": int(match.group(2)),

                    # 训练损失。
                    "train_loss": float(match.group(3)),

                    # 训练准确率百分数。
                    "train_acc": float(match.group(4)),

                    # 验证损失。
                    "val_loss": float(match.group(5)),

                    # 验证准确率百分数。
                    "val_acc": float(match.group(6)),
                }
            )

        # 如果当前日志里确实解析到了记录，就保存这一组 run。
        if records:
            # path.stem 是去掉扩展名的文件名。
            runs.append({"name": path.stem, "records": records})

    # 返回全部训练日志记录。
    return runs


# 定义 checkpoint 评估函数；它会在验证集上重新跑一遍模型。
def evaluate_checkpoint():
    """加载最佳模型，在验证集上计算准确率、分类报告和混淆矩阵。"""
    # 创建完整缓存数据集；如果缺缓存，会自动生成。
    dataset = PrecomputedMelDataset(config.DATA_DIR, config.MEL_CACHE_DIR)

    # 提取全部样本的标签列表。
    labels = [label for _, label in dataset.samples]

    # 使用和训练时相同的分层划分方式得到训练/验证索引。
    train_indices, val_indices = _stratified_split(
        # 全部样本标签。
        labels,

        # 训练集比例。
        config.TRAIN_RATIO,

        # 随机种子。
        config.RANDOM_SEED,
    )

    # 用验证索引创建验证集包装器；验证集只做归一化，不做增强。
    val_dataset = ValMelDataset(dataset, val_indices)

    # 创建验证 DataLoader。
    val_loader = DataLoader(
        # 验证数据集。
        val_dataset,

        # 批量大小沿用训练配置。
        batch_size=config.BATCH_SIZE,

        # 验证不需要打乱。
        shuffle=False,

        # 为了报告评估更稳妥，这里使用 0 个子进程，避免 Windows 多进程问题。
        num_workers=0,

        # 如果 CUDA 可用，就启用固定内存以加速拷贝。
        pin_memory=torch.cuda.is_available(),
    )

    # 如果当前机器有 CUDA，就使用配置设备，否则强制使用 CPU。
    device = config.DEVICE if torch.cuda.is_available() else torch.device("cpu")

    # 加载保存好的最佳模型 checkpoint。
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=False)

    # 根据 checkpoint 中记录的模型结构创建模型；没有 arch 时用配置默认值。
    model = build_model(checkpoint.get("arch", config.MODEL_ARCH)).to(device)

    # 加载模型参数。
    model.load_state_dict(checkpoint["model_state"])

    # 切换到评估模式。
    model.eval()

    # y_true 保存真实标签。
    y_true = []

    # y_pred 保存模型预测标签。
    y_pred = []

    # 评估时不需要梯度，减少显存和计算开销。
    with torch.no_grad():
        # 逐 batch 遍历验证集。
        for mel, target in val_loader:
            # 把 Mel 频谱图移动到设备。
            mel = mel.to(device, non_blocking=True)

            # 前向传播，得到每个类别的 logits。
            logits = model(mel)

            # 取每个样本 logits 最大的类别作为预测结果，并转成 Python 列表。
            pred = logits.argmax(dim=1).cpu().numpy().tolist()

            # 将当前 batch 的预测标签加入总列表。
            y_pred.extend(pred)

            # 将当前 batch 的真实标签加入总列表。
            y_true.extend(target.numpy().tolist())

    # 计算混淆矩阵；行是真实类别，列是预测类别。
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))

    # 计算分类报告，包括 precision、recall、f1-score 等指标。
    report = classification_report(
        # 真实标签。
        y_true,

        # 预测标签。
        y_pred,

        # 固定标签顺序，确保和 CLASSES 一致。
        labels=list(range(config.NUM_CLASSES)),

        # 把类别编号映射成类别名称。
        target_names=config.CLASSES,

        # 输出字典格式，方便保存到 JSON。
        output_dict=True,

        # 遇到除以 0 时返回 0，避免报 warning。
        zero_division=0,
    )

    # 统计每个类别在完整数据集中的样本数量。
    class_counts = {
        # key 是类别名，value 是这个类别的样本数。
        class_name: int(sum(1 for label in labels if label == idx))

        # 遍历类别编号和类别名。
        for idx, class_name in enumerate(config.CLASSES)
    }

    # per_class_acc 保存每个类别自己的准确率，也就是该类别召回率。
    per_class_acc = {}

    # 遍历每个类别编号和名称。
    for idx, class_name in enumerate(config.CLASSES):
        # 当前类别在验证集中真实出现的次数，就是混淆矩阵该行总和。
        row_total = int(cm[idx].sum())

        # 如果该类别有样本，就用对角线正确数 / 行总数；否则记为 0。
        per_class_acc[class_name] = float(cm[idx, idx] / row_total) if row_total else 0.0

    # confusions 保存所有非对角线误分情况。
    confusions = []

    # 遍历真实类别。
    for i, true_name in enumerate(config.CLASSES):
        # 遍历预测类别。
        for j, pred_name in enumerate(config.CLASSES):
            # 只关心真实类别和预测类别不同，并且确实有误分数量的位置。
            if i != j and cm[i, j] > 0:
                # 把这一组误分加入列表。
                confusions.append(
                    {
                        # 真实类别名。
                        "true": true_name,

                        # 预测类别名。
                        "pred": pred_name,

                        # 误分数量。
                        "count": int(cm[i, j]),
                    }
                )

    # 按误分数量从大到小排序，方便报告展示最严重的混淆。
    confusions.sort(key=lambda item: item["count"], reverse=True)

    # 组合最终指标字典。
    metrics = {
        # checkpoint 部分记录模型文件自带的信息。
        "checkpoint": {
            # 保存模型来自第几个 epoch。
            "epoch": int(checkpoint.get("epoch", -1)),

            # 保存模型结构名称。
            "arch": checkpoint.get("arch", config.MODEL_ARCH),

            # checkpoint 里记录的验证准确率。
            "val_acc_saved": float(checkpoint.get("val_acc", 0.0)),

            # checkpoint 里记录的关键配置。
            "config": checkpoint.get("config", {}),
        },

        # evaluation 部分记录本脚本重新评估得到的指标。
        "evaluation": {
            # 验证集总体准确率。
            "accuracy": float(accuracy_score(y_true, y_pred)),

            # 验证集样本数。
            "val_samples": len(y_true),

            # 训练集样本数。
            "train_samples": len(train_indices),

            # 总样本数。
            "total_samples": len(labels),

            # sklearn 生成的详细分类报告。
            "classification_report": report,

            # 每个类别自己的准确率。
            "per_class_acc": per_class_acc,

            # 最明显的前 10 组误分。
            "top_confusions": confusions[:10],
        },

        # 完整数据集中每个类别的样本数量。
        "class_counts": class_counts,

        # 类别名称列表。
        "classes": config.CLASSES,
    }

    # 返回指标字典和混淆矩阵。
    return metrics, cm


# 定义类别分布图绘制函数。
def plot_class_distribution(metrics):
    """绘制每个类别的样本数量柱状图。"""
    # 从指标中读取类别名列表。
    names = metrics["classes"]

    # 按类别名取出对应样本数。
    counts = [metrics["class_counts"][name] for name in names]

    # 创建图像和坐标轴，设置画布大小。
    fig, ax = plt.subplots(figsize=(12, 5.2))

    # 绘制柱状图。
    bars = ax.bar(names, counts, color="#2E74B5")

    # 设置图标题。
    ax.set_title("Dataset Class Distribution", fontsize=14, weight="bold")

    # 设置 y 轴标题。
    ax.set_ylabel("Number of audio clips")

    # 设置 x 轴标题。
    ax.set_xlabel("Class")

    # 设置 y 轴上限，给柱子顶部数字留空间。
    ax.set_ylim(0, max(counts) * 1.15)

    # 设置 x 轴标签旋转 45 度，避免类别名挤在一起。
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)

    # 遍历每个柱子和对应数量。
    for bar, count in zip(bars, counts):
        # 在柱子顶部写出具体数量。
        ax.text(
            # x 坐标放在柱子中间。
            bar.get_x() + bar.get_width() / 2,

            # y 坐标放在柱子上方一点。
            bar.get_height() + 8,

            # 显示样本数。
            str(count),

            # 水平居中。
            ha="center",

            # 垂直方向贴着文字底部。
            va="bottom",

            # 字号较小，避免拥挤。
            fontsize=7,
        )

    # 自动调整布局，减少标签被裁剪的概率。
    fig.tight_layout()

    # 保存类别分布图到 report_assets。
    fig.savefig(ASSET_DIR / "class_distribution.png", dpi=220)

    # 关闭图像，释放内存。
    plt.close(fig)


# 定义训练曲线绘制函数。
def plot_training_curves(runs, metrics):
    """根据训练日志绘制 loss 和 accuracy 曲线。"""
    # 创建上下两个子图：上面画 loss，下面画 accuracy。
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=False)

    # 预设几种颜色，用于不同训练日志。
    colors = ["#2E74B5", "#C0504D", "#70AD47"]

    # 遍历所有训练日志 run。
    for idx, run in enumerate(runs):
        # 取出当前 run 的逐 epoch 记录。
        records = run["records"]

        # 提取 epoch 列表作为横轴。
        epochs = [r["epoch"] for r in records]

        # 给当前 run 生成显示标签。
        label = f"Run {idx + 1}"

        # 按 idx 循环选择颜色。
        color = colors[idx % len(colors)]

        # 在上图画训练损失曲线。
        axes[0].plot(
            # 横轴 epoch。
            epochs,

            # 纵轴训练损失。
            [r["train_loss"] for r in records],

            # 实线表示训练集。
            linestyle="-",

            # 当前 run 的颜色。
            color=color,

            # 图例文字。
            label=f"{label} train loss",
        )

        # 在上图画验证损失曲线。
        axes[0].plot(
            # 横轴 epoch。
            epochs,

            # 纵轴验证损失。
            [r["val_loss"] for r in records],

            # 虚线表示验证集。
            linestyle="--",

            # 使用同一 run 的颜色。
            color=color,

            # 图例文字。
            label=f"{label} val loss",
        )

        # 在下图画训练准确率曲线。
        axes[1].plot(
            # 横轴 epoch。
            epochs,

            # 纵轴训练准确率。
            [r["train_acc"] for r in records],

            # 实线表示训练集。
            linestyle="-",

            # 当前 run 的颜色。
            color=color,

            # 图例文字。
            label=f"{label} train acc",
        )

        # 在下图画验证准确率曲线。
        axes[1].plot(
            # 横轴 epoch。
            epochs,

            # 纵轴验证准确率。
            [r["val_acc"] for r in records],

            # 虚线表示验证集。
            linestyle="--",

            # 当前 run 的颜色。
            color=color,

            # 图例文字。
            label=f"{label} val acc",
        )

    # 从 metrics 中取出 checkpoint 信息。
    ckpt = metrics["checkpoint"]

    # 如果 checkpoint 记录了有效 epoch 和验证准确率，就在图上标星。
    if ckpt["epoch"] > 0 and ckpt["val_acc_saved"] > 0:
        # 在准确率图上画一个星标表示保存的最佳模型。
        axes[1].scatter(
            # 星标横坐标是 checkpoint epoch。
            [ckpt["epoch"]],

            # 星标纵坐标是保存时验证准确率百分数。
            [ckpt["val_acc_saved"] * 100],

            # 使用星形标记。
            marker="*",

            # 设置标记大小。
            s=160,

            # 设置星标颜色。
            color="#8064A2",

            # 设置图例文字。
            label=f"Saved {ckpt['arch']} best: {ckpt['val_acc_saved'] * 100:.2f}%",

            # zorder 让星标显示在曲线之上。
            zorder=5,
        )

    # 设置 loss 子图标题。
    axes[0].set_title("Training and Validation Loss Records", fontsize=13, weight="bold")

    # 设置 loss 子图 y 轴名称。
    axes[0].set_ylabel("Loss")

    # 添加浅色网格，方便读数。
    axes[0].grid(True, alpha=0.25)

    # 添加 loss 子图图例。
    axes[0].legend(fontsize=8, ncol=2)

    # 设置 accuracy 子图标题。
    axes[1].set_title("Training and Validation Accuracy Records", fontsize=13, weight="bold")

    # 设置 accuracy 子图 x 轴名称。
    axes[1].set_xlabel("Epoch")

    # 设置 accuracy 子图 y 轴名称。
    axes[1].set_ylabel("Accuracy (%)")

    # 准确率范围固定为 0 到 105，给 100% 上方留一点空间。
    axes[1].set_ylim(0, 105)

    # 添加浅色网格。
    axes[1].grid(True, alpha=0.25)

    # 添加 accuracy 子图图例。
    axes[1].legend(fontsize=8, ncol=2)

    # 自动调整布局。
    fig.tight_layout()

    # 保存训练曲线图。
    fig.savefig(ASSET_DIR / "training_curves.png", dpi=220)

    # 关闭图像，释放内存。
    plt.close(fig)


# 定义混淆矩阵绘制函数。
def plot_confusion_matrix(metrics, cm):
    """绘制验证集混淆矩阵热力图。"""
    # 读取类别名列表。
    names = metrics["classes"]

    # 创建图像和坐标轴。
    fig, ax = plt.subplots(figsize=(13, 11))

    # 用 Blues 色图显示混淆矩阵。
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # 添加颜色条，颜色越深表示数量越大。
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 设置坐标轴刻度、标签和标题。
    ax.set(
        # x 轴刻度位置。
        xticks=np.arange(len(names)),

        # y 轴刻度位置。
        yticks=np.arange(len(names)),

        # x 轴类别名。
        xticklabels=names,

        # y 轴类别名。
        yticklabels=names,

        # y 轴说明。
        ylabel="True label",

        # x 轴说明。
        xlabel="Predicted label",

        # 图标题。
        title="Confusion Matrix on Validation Set",
    )

    # 旋转 x 轴标签，防止重叠。
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)

    # 设置 y 轴标签字号。
    ax.tick_params(axis="y", labelsize=8)

    # 计算颜色阈值，用来决定矩阵数字显示黑色还是白色。
    threshold = cm.max() / 2.0 if cm.size else 0

    # 遍历混淆矩阵每一行。
    for i in range(cm.shape[0]):
        # 遍历混淆矩阵每一列。
        for j in range(cm.shape[1]):
            # 当前格子的样本数量。
            value = int(cm[i, j])

            # 只在数量非 0 的格子里写数字，减少视觉干扰。
            if value:
                # 在热力图格子中间写入数量。
                ax.text(
                    # x 坐标是列号。
                    j,

                    # y 坐标是行号。
                    i,

                    # 显示数量。
                    str(value),

                    # 水平居中。
                    ha="center",

                    # 垂直居中。
                    va="center",

                    # 深色背景用白字，浅色背景用黑字。
                    color="white" if value > threshold else "black",

                    # 字号较小，适合 20x20 矩阵。
                    fontsize=6,
                )

    # 自动调整布局，减少标签裁剪。
    fig.tight_layout()

    # 保存混淆矩阵图。
    fig.savefig(ASSET_DIR / "confusion_matrix.png", dpi=220)

    # 关闭图像，释放内存。
    plt.close(fig)


# 定义模型结构示意图绘制函数。
def plot_model_architecture():
    """绘制 cnn_v2 从输入到输出的模块流程图。"""
    # blocks 保存每个结构块的标题和说明。
    blocks = [
        # 输入块。
        ("Input", "1 x 128 x T\nMel spectrogram"),

        # 第一阶段卷积块。
        ("Stage 1", "DoubleConv 1->32\nBN + SiLU + Pool"),

        # 第二阶段卷积块。
        ("Stage 2", "DoubleConv 32->64\nBN + SiLU + Pool"),

        # 第三阶段卷积块。
        ("Stage 3", "DoubleConv 64->128\nDropout2d + Pool"),

        # 第四阶段卷积块。
        ("Stage 4", "DoubleConv 128->256\nDropout2d + Pool"),

        # 全局平均池化块。
        ("GAP", "AdaptiveAvgPool2d\n1 x 1"),

        # 分类器块。
        ("Classifier", "Dropout + FC 256\nFC -> 20 classes"),
    ]

    # 创建结构图画布。
    fig, ax = plt.subplots(figsize=(12, 3.8))

    # 关闭坐标轴，因为这里画的是流程图不是数据图。
    ax.axis("off")

    # 为每个模块生成均匀分布的 x 坐标。
    x_positions = np.linspace(0.035, 0.865, len(blocks))

    # 设置每个矩形块的宽度。
    width = 0.105

    # 同时遍历模块信息和 x 坐标。
    for idx, ((title, body), x) in enumerate(zip(blocks, x_positions)):
        # 创建一个矩形表示一个模型模块。
        rect = plt.Rectangle(
            # 矩形左下角坐标。
            (x, 0.35),

            # 矩形宽度。
            width,

            # 矩形高度。
            0.38,

            # 边框宽度。
            linewidth=1.3,

            # 边框颜色。
            edgecolor="#1F4D78",

            # 第一个输入块用浅灰，其余模块用浅蓝。
            facecolor="#E8EEF5" if idx else "#F2F4F7",
        )

        # 把矩形添加到坐标轴。
        ax.add_patch(rect)

        # 在矩形上半部分写模块标题。
        ax.text(x + width / 2, 0.62, title, ha="center", va="center", fontsize=10, weight="bold")

        # 在矩形下半部分写模块说明。
        ax.text(x + width / 2, 0.47, body, ha="center", va="center", fontsize=7.5)

        # 如果不是最后一个模块，就画箭头指向下一个模块。
        if idx < len(blocks) - 1:
            # 添加箭头。
            ax.annotate(
                # 箭头文字为空，只显示箭头本身。
                "",

                # 箭头终点，靠近下一个矩形左侧。
                xy=(x_positions[idx + 1] - 0.008, 0.54),

                # 箭头起点，靠近当前矩形右侧。
                xytext=(x + width + 0.008, 0.54),

                # 设置箭头样式、线宽和颜色。
                arrowprops=dict(arrowstyle="->", lw=1.2, color="#1F4D78"),
            )

    # 设置整张图标题。
    ax.set_title("CNN V2 Model Architecture", fontsize=14, weight="bold", pad=12)

    # 自动调整布局。
    fig.tight_layout()

    # 保存模型结构图。
    fig.savefig(ASSET_DIR / "model_architecture.png", dpi=220)

    # 关闭图像，释放内存。
    plt.close(fig)


# 定义主函数；按顺序生成所有报告素材。
def main():
    # 解析训练日志，得到历史训练曲线数据。
    runs = parse_training_logs()

    # 评估 checkpoint，得到指标字典和混淆矩阵。
    metrics, cm = evaluate_checkpoint()

    # 把训练日志摘要加入 metrics，方便报告或后续分析引用。
    metrics["training_log_runs"] = [
        {
            # 训练日志文件名。
            "name": run["name"],

            # 该日志记录了多少个 epoch。
            "epochs": len(run["records"]),

            # 该日志中最好的验证准确率。
            "best_val_acc": max(r["val_acc"] for r in run["records"]),

            # 该日志中验证准确率最高的 epoch。
            "best_epoch": max(run["records"], key=lambda r: r["val_acc"])["epoch"],
        }

        # 遍历所有解析出的训练日志。
        for run in runs
    ]

    # 绘制类别分布图。
    plot_class_distribution(metrics)

    # 绘制训练曲线图。
    plot_training_curves(runs, metrics)

    # 绘制混淆矩阵图。
    plot_confusion_matrix(metrics, cm)

    # 绘制模型结构示意图。
    plot_model_architecture()

    # 将指标字典写入 metrics.json。
    (ASSET_DIR / "metrics.json").write_text(
        # ensure_ascii=False 保留中文，indent=2 让 JSON 更好读。
        json.dumps(metrics, ensure_ascii=False, indent=2),

        # 使用 UTF-8 编码保存。
        encoding="utf-8",
    )

    # 在终端打印指标 JSON，方便用户直接查看。
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


# 只有直接运行 python generate_report_assets.py 时才执行主函数。
if __name__ == "__main__":
    # 调用主函数，生成全部报告素材。
    main()
