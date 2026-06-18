# predict.py
# ============================================================
# 推理脚本：输入一个音频文件，输出预测类别 + 置信度
# ============================================================
#
# 训练是“教模型”，预测是“让训练好的模型答题”。
# 这个文件会加载 best_model.pth，然后把用户给的一段音频处理成和训练时一样的 Mel 频谱图，
# 最后输出模型认为最可能的前 5 个类别。

# 导入 sys，用来读取命令行参数，例如 python predict.py xxx.wav 中的 xxx.wav。
import sys

# 导入 PyTorch；推理时需要加载模型、处理张量和计算 softmax。
import torch

# 导入 torchaudio；用于读取用户传入的音频文件。
import torchaudio

# 导入 torchaudio.transforms，并起别名 T；用于重采样和提取 Mel 频谱图。
import torchaudio.transforms as T

# 导入配置文件中的路径、音频参数、类别列表和设备等。
from config import *

# 导入模型构建函数；会根据 checkpoint 中记录的结构名创建对应模型。
from model import build_model


# 定义模型加载函数；checkpoint_path 是模型文件路径，device 是加载到 CPU 还是 GPU。
def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """从 checkpoint 加载训练好的模型，并切换到评估模式。"""
    # torch.load 读取训练时保存的 checkpoint；map_location 保证能加载到当前设备。
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 从 checkpoint 中读取模型结构名；如果旧模型没有 arch 字段，就默认 cnn_v1。
    arch = ckpt.get("arch", "cnn_v1")

    # 按结构名创建一个空模型，再移动到目标设备。
    model = build_model(arch).to(device)

    # 把 checkpoint 中保存的模型参数加载到模型里。
    model.load_state_dict(ckpt["model_state"])

    # 切换到评估模式；推理时会关闭 Dropout，并使用 BatchNorm 的固定统计量。
    model.eval()

    # 打印模型来源、结构、保存轮次和当时验证准确率，方便确认用的是哪个模型。
    print(f"[模型] 加载自 {checkpoint_path}  (arch={arch}, epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']*100:.2f}%)")

    # 返回加载好参数的模型。
    return model


# 定义单条音频预处理函数；它必须尽量和训练阶段保持一致。
def preprocess(audio_path: str, device: torch.device) -> torch.Tensor:
    """把用户输入的音频文件转换成模型需要的 4 维输入张量。"""
    # 计算目标音频长度对应的采样点数量。
    n_samples = int(SAMPLE_RATE * DURATION)

    # 从硬盘读取音频文件，得到 waveform 波形和 sr 原始采样率。
    waveform, sr = torchaudio.load(audio_path)

    # 如果音频是双声道或多声道，就平均成单声道。
    if waveform.shape[0] > 1:
        # 保持输出形状为 (1, 时间点数)，与训练数据格式一致。
        waveform = waveform.mean(dim=0, keepdim=True)

    # 如果采样率和训练配置不一致，就重采样到训练时使用的 SAMPLE_RATE。
    if sr != SAMPLE_RATE:
        # 重采样保证同一段声音在时间尺度上和训练数据一致。
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

    # 读取当前音频实际采样点数量。
    n = waveform.shape[1]

    # 如果音频比目标长度长，就截断到前 n_samples 个采样点。
    if n >= n_samples:
        # 截断后每条音频长度固定，模型输入尺寸稳定。
        waveform = waveform[:, :n_samples]

    # 如果音频比目标长度短，就在末尾补零。
    else:
        # 补零不会增加真实声音，只是让张量长度达到模型要求。
        waveform = torch.nn.functional.pad(waveform, (0, n_samples - n))

    # 创建 MelSpectrogram 转换器，参数必须和训练时保持一致。
    mel = T.MelSpectrogram(
        # 使用统一采样率。
        sample_rate=SAMPLE_RATE,

        # 使用统一 FFT 点数。
        n_fft=N_FFT,

        # 使用统一帧移。
        hop_length=HOP_LENGTH,

        # 使用统一 Mel 频带数量。
        n_mels=N_MELS,

        # 使用统一最低频率。
        f_min=F_MIN,

        # 使用统一最高频率。
        f_max=F_MAX,

    # 立即对 waveform 执行 Mel 频谱转换。
    )(waveform)

    # 将 Mel 幅度转换为 dB 尺度，和训练缓存中的特征保持一致。
    mel = T.AmplitudeToDB(top_db=80)(mel)

    # 计算当前 Mel 图最小值。
    mel_min = mel.min()

    # 计算当前 Mel 图最大值。
    mel_max = mel.max()

    # 如果最大值大于最小值，就执行 [0, 1] 归一化。
    if mel_max > mel_min:
        # 归一化后数值范围和训练/验证阶段一致。
        mel = (mel - mel_min) / (mel_max - mel_min)

    # unsqueeze(0) 增加 batch 维度，把 (1, 128, T) 变成 (1, 1, 128, T)。
    return mel.unsqueeze(0).to(device)


# 推理阶段不需要梯度；这个装饰器可以减少显存占用并加快运行。
@torch.no_grad()
# 定义预测函数；audio_path 是待预测音频，model_path 默认使用配置中的最佳模型。
def predict(audio_path: str, model_path: str = MODEL_SAVE_PATH):
    """加载模型、预处理音频、输出 Top-5 预测类别和置信度。"""
    # 加载训练好的模型到 DEVICE。
    model = load_model(model_path, DEVICE)

    # 把输入音频预处理成模型需要的张量。
    tensor = preprocess(audio_path, DEVICE)

    # 将音频张量送入模型，得到每个类别的 logits。
    logits = model(tensor)

    # softmax 把 logits 转成概率；[0] 取出 batch 中唯一一个样本的概率向量。
    probs = torch.softmax(logits, dim=1)[0]

    # 取概率最高的前 5 个类别及其概率。
    top5_probs, top5_idx = probs.topk(5)

    # 打印被预测的音频路径。
    print(f"\n音频文件: {audio_path}")

    # 打印表格分隔线。
    print("-" * 40)

    # 打印表头：排名、类别、置信度。
    print(f"{'排名':<4} {'类别':<18} {'置信度':>8}")

    # 打印表格分隔线。
    print("-" * 40)

    # 遍历 Top-5 结果；enumerate(..., 1) 让排名从 1 开始。
    for rank, (idx, prob) in enumerate(zip(top5_idx, top5_probs), 1):
        # idx 是类别编号，prob 是该类别概率；item() 把张量数值转成 Python 浮点数。
        print(f"{rank:<4} {CLASSES[idx]:<18} {prob.item()*100:>7.2f}%")

    # 打印表格底部分隔线。
    print("-" * 40)

    # 打印最终预测结果，也就是 Top-1 类别。
    print(f"预测结果: {CLASSES[top5_idx[0]]}  ({top5_probs[0].item()*100:.2f}%)\n")

    # 返回最可能类别名和它的概率，方便别的 Python 代码调用这个函数。
    return CLASSES[top5_idx[0]], top5_probs[0].item()


# 只有直接运行 python predict.py 时，下面的命令行入口才会执行。
if __name__ == "__main__":
    # 如果命令行参数少于 2 个，说明用户没有提供音频路径。
    if len(sys.argv) < 2:
        # 打印正确用法。
        print("用法: python predict.py <音频文件路径>")

        # 打印示例，帮助用户照着改路径。
        print("示例: python predict.py D:/test/chips_sample.wav")

        # 用状态码 1 退出，表示命令使用不正确。
        sys.exit(1)

    # 读取命令行中的第一个用户参数，作为待预测音频路径。
    audio_file = sys.argv[1]

    # 调用预测函数并打印结果。
    predict(audio_file)
