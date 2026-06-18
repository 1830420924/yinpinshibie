# train.py
# ============================================================
# 训练入口：支持 CUDA GPU 加速 + AMP 混合精度 + 早停 + 学习率调度
# ============================================================
#
# 给大一新生的总览：
# 这个文件负责“教模型听懂进食声音”。它会先读取数据，再建立 CNN 模型，
# 然后反复执行“训练一轮 -> 验证一轮 -> 根据验证结果保存最好模型”的流程。
# 可以把训练想象成学生刷题：训练集是练习题，验证集是模拟考试，
# loss 是错题程度，accuracy 是答对比例，optimizer 是改正方法，
# scheduler 是学习率节奏表，early stopping 是发现成绩不涨就及时停下。

# 导入 time，用来统计每个 epoch 训练和验证一共花了多少秒。
import time

# 导入 PyTorch 主库；模型、张量、GPU 操作都依赖它。
import torch

# 导入 torch.nn，并起别名 nn；里面包含损失函数、神经网络层等常用模块。
import torch.nn as nn

# 导入 torch.optim，并起别名 optim；里面包含 AdamW 等优化器。
import torch.optim as optim

# 导入余弦退火学习率调度器，让学习率像余弦曲线一样逐渐变小。
from torch.optim.lr_scheduler import CosineAnnealingLR

# 导入 AMP 混合精度工具：autocast 自动使用半精度，GradScaler 防止梯度太小。
from torch.amp import autocast, GradScaler

# 导入 config.py 中的所有配置，例如路径、类别数、训练轮数、学习率、设备等。
from config import *

# 导入数据加载函数；它会返回训练集 DataLoader、验证集 DataLoader 和类别权重。
from dataset import get_dataloaders

# 导入模型构建函数和参数统计函数。
from model import build_model, count_params


# ────────────────────────────────────────────────
# 工具函数：训练开始前和每一轮训练/验证都会用到
# ────────────────────────────────────────────────


# 定义随机种子设置函数；seed 是随机种子的整数值。
def seed_everything(seed: int):
    """固定主要随机源，让同一份代码更容易复现实验结果。"""
    # 固定 PyTorch 在 CPU 上产生随机数的顺序，例如随机打乱数据时会用到。
    torch.manual_seed(seed)

    # 判断当前电脑是否可以使用 CUDA GPU。
    if torch.cuda.is_available():
        # 固定所有 CUDA GPU 上的随机种子，避免 GPU 训练时每次结果差太远。
        torch.cuda.manual_seed_all(seed)

    # 让 cuDNN 自动寻找更快的卷积算法；输入形状稳定时通常可以加速训练。
    torch.backends.cudnn.benchmark = True


# 定义“训练一个 epoch”的函数；一个 epoch 表示模型完整看一遍训练集。
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """完成一轮训练，并返回这一轮的平均损失和训练准确率。"""
    # 将模型切换到训练模式；Dropout、BatchNorm 等层会按训练规则工作。
    model.train()

    # total_loss 累计所有样本的损失，correct 累计预测正确数量，total 累计样本总数。
    total_loss, correct, total = 0.0, 0, 0

    # 只有在 CUDA GPU 上才启用 AMP；CPU 上启用混合精度收益不大且可能不兼容。
    amp_enabled = device.type == "cuda"

    # 从 DataLoader 中一批一批取数据；mel 是 Mel 频谱图，labels 是正确类别编号。
    for mel, labels in loader:
        # 把输入特征和标签移动到指定设备；non_blocking=True 可在固定内存下加速拷贝。
        mel, labels = mel.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # 清空上一批数据留下的梯度；set_to_none=True 通常更省内存也更快。
        optimizer.zero_grad(set_to_none=True)

        # autocast 会在合适的计算中使用半精度，从而减少显存占用并提高速度。
        with autocast(device_type=device.type, enabled=amp_enabled):
            # 前向传播：把 Mel 频谱图送进模型，得到每个类别的原始分数 logits。
            outputs = model(mel)

            # 用损失函数比较模型输出和真实标签；loss 越小表示预测越接近答案。
            loss = criterion(outputs, labels)

        # scaler.scale(loss) 会放大 loss 再反向传播，防止半精度训练时梯度下溢。
        scaler.scale(loss).backward()

        # 根据刚算出的梯度更新模型参数；这一步就是“模型学习”的核心动作。
        scaler.step(optimizer)

        # 更新 scaler 内部的缩放倍率，让后续 batch 的混合精度训练保持稳定。
        scaler.update()

        # loss.item() 是当前 batch 的平均损失，乘 batch 大小后变成样本级累计损失。
        total_loss += loss.item() * labels.size(0)

        # argmax(dim=1) 取每个样本分数最高的类别，作为模型预测结果。
        preds = outputs.argmax(dim=1)

        # 统计本 batch 中预测类别等于真实类别的样本数量。
        correct += (preds == labels).sum().item()

        # 统计已经处理过的样本数量；labels.size(0) 就是当前 batch 的样本数。
        total += labels.size(0)

    # 返回平均损失和准确率；平均损失=总损失/总样本数，准确率=正确数/总样本数。
    return total_loss / total, correct / total


# 这个装饰器表示 evaluate 内部不需要计算梯度；验证阶段只看效果，不更新模型。
@torch.no_grad()
# 定义验证函数；它和训练函数很像，但不会 backward，也不会 optimizer.step。
def evaluate(model, loader, criterion, device):
    """在验证集上评估模型，并返回平均验证损失和验证准确率。"""
    # 切换到评估模式；Dropout 会关闭，BatchNorm 会使用训练时记录的统计量。
    model.eval()

    # 初始化验证集上的累计损失、预测正确数和样本总数。
    total_loss, correct, total = 0.0, 0, 0

    # 和训练一样，只在 CUDA GPU 上启用 AMP 混合精度推理。
    amp_enabled = device.type == "cuda"

    # 逐批读取验证集；验证集不做随机增强，用来模拟模型面对“没见过题”的表现。
    for mel, labels in loader:
        # 将验证数据移动到 CPU 或 GPU，与模型所在设备保持一致。
        mel, labels = mel.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # 验证阶段也可以使用 autocast 加速，但不会计算梯度。
        with autocast(device_type=device.type, enabled=amp_enabled):
            # 前向传播，得到每个样本属于各个类别的原始分数。
            outputs = model(mel)

            # 计算验证损失，用来观察模型是否在没见过的数据上也表现稳定。
            loss = criterion(outputs, labels)

        # 累加验证损失；乘 batch 大小是为了最后按样本数求平均。
        total_loss += loss.item() * labels.size(0)

        # 取分数最高的类别作为预测类别。
        preds = outputs.argmax(dim=1)

        # 累加预测正确的样本数量。
        correct += (preds == labels).sum().item()

        # 累加验证样本数量。
        total += labels.size(0)

    # 返回验证集平均损失和验证准确率。
    return total_loss / total, correct / total


# ────────────────────────────────────────────────
# 主训练流程：从这里把数据、模型、损失函数、优化器和训练循环串起来
# ────────────────────────────────────────────────


# 定义 main 函数；把训练流程放进函数里，方便脚本入口调用。
def main():
    """项目训练入口：准备数据与模型，并执行完整训练流程。"""
    # 固定随机种子，尽量让数据划分、数据打乱和随机增强可复现。
    seed_everything(RANDOM_SEED)

    # 打印分隔线，让终端输出更容易阅读。
    print("=" * 60)

    # 打印当前使用的设备，例如 cpu 或 cuda。
    print(f"  设备: {DEVICE}")

    # 如果设备是 CUDA，就继续打印显卡名称和显存信息。
    if DEVICE.type == "cuda":
        # 打印第 0 块 GPU 的名称，例如 NVIDIA GeForce RTX 3060 Laptop GPU。
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

        # 打印第 0 块 GPU 的总显存，除以 1e9 后约等于 GB。
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 打印 AMP 是否启用；只有 CUDA 训练时才启用。
    print(f"  AMP : {'已启用' if DEVICE.type == 'cuda' else '未启用'}")

    # 打印本次训练选择的模型结构，例如 cnn_v1 或 cnn_v2。
    print(f"  模型: {MODEL_ARCH}")

    # 再打印一条分隔线，结束环境信息区域。
    print("=" * 60)

    # 第 1 步：构建数据加载器；训练集用于学习，验证集用于检查泛化效果。
    train_loader, val_loader, class_weights = get_dataloaders(DATA_DIR, MEL_CACHE_DIR)

    # 第 2 步：根据配置创建模型，并把模型参数移动到 DEVICE 指定的设备。
    model = build_model(MODEL_ARCH).to(DEVICE)

    # 打印模型参数量；参数越多模型容量越大，但也更容易过拟合、训练更慢。
    print(count_params(model))

    # 第 3 步：创建损失函数；CrossEntropyLoss 常用于多分类任务。
    criterion = nn.CrossEntropyLoss(
        # 类别权重用于处理类别数量不平衡；样本少的类别会被适当提高损失权重。
        weight=class_weights.to(DEVICE),

        # label_smoothing 会让标签不那么“绝对”，有助于减轻过拟合和过度自信。
        label_smoothing=LABEL_SMOOTHING,
    )

    # 创建 AdamW 优化器；它负责根据梯度更新模型参数，并带有权重衰减正则化。
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 创建余弦退火学习率调度器；训练前期步子大，后期学习率逐渐变小更细调。
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 创建 GradScaler；CUDA 下启用，CPU 下自动关闭，配合 AMP 保证训练稳定。
    scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")

    # best_val_acc 保存目前见过的最高验证准确率，用来判断是否要保存模型。
    best_val_acc = 0.0

    # no_improve 记录验证准确率连续多少个 epoch 没有提升，用于早停。
    no_improve = 0

    # history 用列表记录每一轮的训练/验证损失和准确率，方便后续画曲线。
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # 打印训练开始信息，告诉用户计划训练多少个 epoch。
    print(f"\n开始训练，共 {NUM_EPOCHS} 个 epoch\n")

    # 从第 1 轮训练到第 NUM_EPOCHS 轮；range 右边界不包含，所以要 +1。
    for epoch in range(1, NUM_EPOCHS + 1):
        # 记录本轮开始时间，后面用来计算本轮耗时。
        t0 = time.time()

        # 在训练集上跑一轮：模型会前向传播、反向传播，并更新参数。
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)

        # 在验证集上评估一轮：只计算指标，不更新参数。
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        # 让学习率调度器前进一步，更新下一轮训练要使用的学习率。
        scheduler.step()

        # 计算本轮训练和验证总耗时。
        elapsed = time.time() - t0

        # 读取当前优化器中的最新学习率；get_last_lr 返回列表，这里取第一个参数组。
        lr_now = scheduler.get_last_lr()[0]

        # 记录当前 epoch 的训练损失。
        history["train_loss"].append(train_loss)

        # 记录当前 epoch 的验证损失。
        history["val_loss"].append(val_loss)

        # 记录当前 epoch 的训练准确率。
        history["train_acc"].append(train_acc)

        # 记录当前 epoch 的验证准确率。
        history["val_acc"].append(val_acc)

        # 打印本轮的核心指标，方便边训练边观察模型是否进步。
        print(
            # 显示当前是第几轮以及总轮数；03d 表示用 3 位数字显示，不足补 0。
            f"Epoch [{epoch:03d}/{NUM_EPOCHS}] "
            # 显示当前学习率；.2e 表示科学计数法并保留两位小数。
            f"lr={lr_now:.2e}  "
            # 显示训练集损失和训练准确率；准确率乘 100 后变成百分比。
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
            # 显示验证集损失和验证准确率，这是判断模型泛化能力的重点。
            f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
            # 显示本轮耗时，帮助判断训练速度是否正常。
            f"({elapsed:.1f}s)"
        )

        # 如果当前验证准确率超过历史最好成绩，就保存这一版模型。
        if val_acc > best_val_acc:
            # 更新历史最好验证准确率。
            best_val_acc = val_acc

            # 保存 checkpoint；里面不仅有模型参数，也有类别名和关键配置，便于预测脚本加载。
            torch.save({
                # 保存最佳模型来自第几个 epoch，方便之后追踪训练过程。
                "epoch": epoch,

                # 保存模型结构名称，预测时可以自动构建同样的网络。
                "arch": MODEL_ARCH,

                # 保存模型的全部可学习参数，这是 checkpoint 最核心的内容。
                "model_state": model.state_dict(),

                # 保存优化器状态；如果以后想继续训练，可以恢复优化器动量等信息。
                "optimizer_state": optimizer.state_dict(),

                # 保存当时的验证准确率，方便加载模型时知道它的表现。
                "val_acc": val_acc,

                # 保存类别列表，避免未来类别顺序改变导致预测结果解释错误。
                "classes": CLASSES,

                # 保存关键配置，让别人看到 checkpoint 时能知道音频预处理和模型设置。
                "config": {
                    # 训练时使用的采样率。
                    "sample_rate": SAMPLE_RATE,

                    # 每条音频被统一处理成多少秒。
                    "duration": DURATION,

                    # Mel 频谱图的 Mel 频带数量。
                    "n_mels": N_MELS,

                    # 损失函数使用的标签平滑系数。
                    "label_smoothing": LABEL_SMOOTHING,

                    # 模型分类器使用的 Dropout 概率。
                    "dropout": DROPOUT,
                },
            }, MODEL_SAVE_PATH)

            # 打印保存成功的信息，让用户知道最好模型文件写到了哪里。
            print(f"  -> 保存最佳模型 (val_acc={val_acc*100:.2f}%) -> {MODEL_SAVE_PATH}")

            # 既然本轮有提升，就把“连续未提升”计数清零。
            no_improve = 0

        # 如果当前验证准确率没有超过历史最好成绩，就进入未提升逻辑。
        else:
            # 连续未提升轮数加 1。
            no_improve += 1

            # 如果连续未提升次数达到耐心值，就触发早停。
            if no_improve >= EARLY_STOP_PATIENCE:
                # 打印早停原因，说明验证准确率已经很久没变好。
                print(f"\n早停：验证准确率已连续 {EARLY_STOP_PATIENCE} 个 epoch 未提升，停止训练。")

                # 跳出 epoch 循环，提前结束训练。
                break

    # 训练循环结束后，打印本次训练的最佳验证准确率。
    print(f"\n训练完成！最佳验证准确率: {best_val_acc*100:.2f}%")

    # 返回 history，方便其他脚本导入 main 时继续使用训练曲线数据。
    return history


# 只有直接运行 python train.py 时，下面这段入口代码才会执行。
if __name__ == "__main__":
    # 导入 multiprocessing；Windows 上 DataLoader 多进程需要这个模块配合。
    import multiprocessing

    # Windows 打包或多进程启动时需要 freeze_support，避免子进程无限递归启动。
    multiprocessing.freeze_support()

    # 调用主函数，正式开始训练。
    main()
