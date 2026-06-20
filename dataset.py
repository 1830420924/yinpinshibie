# dataset.py
# ============================================================
# 音频数据集 + Mel 频谱图提取（预计算缓存 + 在线增强）
# ============================================================
#
# 这个文件负责把“硬盘里的音频文件”变成“模型能吃进去的张量”。
# 核心流程是：读取音频 -> 统一单声道和采样率 -> 统一 3 秒长度 ->
# 转成 Mel 频谱图 -> 训练集加随机增强 -> 归一化到 [0, 1]。

# 导入 os，用来拼接路径、判断文件夹是否存在、遍历目录等。
import os

# 导入 PyTorch；这里主要用于张量处理、保存/读取缓存和随机划分。
import torch

# 导入 torchaudio，用来读取 wav/mp3/flac/ogg 等音频文件。
import torchaudio

# 导入 torchaudio.transforms，并起别名 T；Mel 频谱、重采样、遮挡增强都在这里。
import torchaudio.transforms as T

# 导入 Dataset 和 DataLoader；Dataset 定义“怎么取一个样本”，DataLoader 负责批量加载。
from torch.utils.data import Dataset, DataLoader

# 导入所有配置项，例如采样率、类别列表、batch size、增强参数等。
from config import *


# ────────────────────────────────────────────────
# 工具函数：把原始音频波形转换成 dB Mel 频谱图
# ────────────────────────────────────────────────


# 定义音频预处理函数；waveform 是音频波形张量，sr 是原始采样率。
def _audio_to_mel_db(waveform, sr):
    """将原始波形转为 dB 尺度的 Mel 频谱图。"""
    # 如果音频有多个声道，例如左右声道，就把它们平均成单声道。
    if waveform.shape[0] > 1:
        # dim=0 表示沿声道维度求平均，keepdim=True 保持形状仍是 (1, 时间点数)。
        waveform = waveform.mean(dim=0, keepdim=True)

    # 如果音频原始采样率不是项目统一采样率，就重采样到 SAMPLE_RATE。
    if sr != SAMPLE_RATE:
        # T.Resample 会把波形从原采样率 sr 转换到目标采样率 SAMPLE_RATE。
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

    # 计算统一时长对应的采样点数量，例如 22050 Hz * 3 秒 = 66150 点。
    n_samples = int(SAMPLE_RATE * DURATION)

    # 读取当前音频实际拥有的采样点数量。
    n = waveform.shape[1]

    # 如果音频长度大于等于目标长度，就截取前 n_samples 个点。
    if n >= n_samples:
        # 这样每条音频输入模型前长度一致，便于组成 batch。
        waveform = waveform[:, :n_samples]

    # 如果音频长度不足目标长度，就进入补零逻辑。
    else:
        # 在音频末尾补 0，让短音频也达到统一长度。
        waveform = torch.nn.functional.pad(waveform, (0, n_samples - n))

    # 创建 MelSpectrogram 转换器，将一维声音波形转换成二维 Mel 频谱图。
    mel = T.MelSpectrogram(
        # 使用项目统一采样率。
        sample_rate=SAMPLE_RATE,

        # n_fft 控制每个短时窗口做傅里叶变换时使用多少采样点。
        n_fft=N_FFT,

        # hop_length 控制相邻窗口之间移动多少采样点。
        hop_length=HOP_LENGTH,

        # n_mels 控制 Mel 频谱图纵轴有多少个 Mel 频带。
        n_mels=N_MELS,

        # f_min 是保留频率范围的下限。
        f_min=F_MIN,

        # f_max 是保留频率范围的上限。
        f_max=F_MAX,

    # 直接调用转换器，把 waveform 转换成 Mel 频谱张量。
    )(waveform)

    # 将幅度 Mel 频谱转换成 dB 尺度，更接近人耳对响度的感知方式。
    mel = T.AmplitudeToDB(top_db=80)(mel)

    # 返回形状大致为 (1, N_MELS, 时间帧数) 的 Mel 频谱图。
    return mel


# ────────────────────────────────────────────────
# 预计算 Mel 频谱缓存数据集：第一次慢，之后训练会快很多
# ────────────────────────────────────────────────


# 定义预计算缓存数据集；继承 Dataset 后可以被 DataLoader 使用。
class PrecomputedMelDataset(Dataset):
    """首次运行时将所有音频预计算为 dB Mel 频谱并缓存到 .pt 文件。
    后续运行直接从缓存加载，大幅加速数据读取。
    返回未经归一化的 dB 频谱，归一化由训练/验证包装器处理。
    """

    # 初始化数据集；data_dir 是原始音频目录，cache_dir 是 Mel 缓存目录。
    def __init__(self, data_dir: str, cache_dir: str):
        # 保存缓存根目录，后面生成 .pt 文件路径时会用到。
        self.cache_dir = cache_dir

        # samples 最终保存可用样本列表，每个元素是 (缓存文件路径, 类别编号)。
        self.samples = []

        # audio_files 先保存扫描到的原始音频、目标缓存路径和标签。
        audio_files = []

        # 遍历所有类别；enumerate 会同时给出类别编号 label_idx 和类别名 class_name。
        for label_idx, class_name in enumerate(CLASSES):
            # 拼出当前类别的原始音频文件夹路径。
            class_audio_dir = os.path.join(data_dir, class_name)

            # 拼出当前类别的缓存文件夹路径。
            class_cache_dir = os.path.join(cache_dir, class_name)

            # 如果某个类别文件夹不存在，就打印警告并跳过该类别。
            if not os.path.isdir(class_audio_dir):
                # 这里不直接报错，是为了允许数据集暂时缺少某些类别文件夹。
                print(f"[警告] 找不到文件夹: {class_audio_dir}")

                # continue 表示跳到下一个类别。
                continue

            # 遍历当前类别文件夹中的文件名，并排序保证扫描顺序稳定。
            for fname in sorted(os.listdir(class_audio_dir)):
                # 只处理常见音频后缀；lower() 可以兼容大写后缀。
                if fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    # 拼出原始音频完整路径。
                    audio_path = os.path.join(class_audio_dir, fname)

                    # 去掉文件扩展名，作为缓存文件名基础。
                    base_name = os.path.splitext(fname)[0]

                    # 拼出对应的缓存 .pt 文件路径。
                    cache_path = os.path.join(class_cache_dir, base_name + ".pt")

                    # 把原始音频路径、缓存路径和类别编号保存到临时列表。
                    audio_files.append((audio_path, cache_path, label_idx))

        # 找出还没有缓存文件的音频；这些文件需要执行 Mel 频谱预计算。
        missing = [(ap, cp, li) for ap, cp, li in audio_files if not os.path.exists(cp)]

        # 如果 missing 不为空，就开始为缺失样本生成缓存。
        if missing:
            # 打印需要处理的音频数量，方便用户知道第一次运行为什么较慢。
            print(f"[预计算] 正在为 {len(missing)} 个音频生成 Mel 频谱缓存...")

            # 逐个处理缺失缓存的音频文件。
            for i, (audio_path, cache_path, label_idx) in enumerate(missing):
                # 确保目标缓存文件夹存在；exist_ok=True 表示已经存在也不报错。
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)

                # try 用来捕获坏音频或读取失败，避免一个文件坏了就中断全局训练。
                try:
                    # 用 torchaudio 读取原始音频，得到波形和采样率。
                    waveform, sr = torchaudio.load(audio_path)

                    # 将原始波形转换成 dB Mel 频谱图。
                    mel = _audio_to_mel_db(waveform, sr)

                    # 把 Mel 张量保存成 .pt 缓存文件，后续可直接加载。
                    torch.save(mel, cache_path)

                # 如果读取或转换失败，就打印错误并跳过当前音频。
                except Exception as e:
                    # 输出被跳过的文件路径和异常信息，方便后面排查数据问题。
                    print(f"  跳过 {audio_path}: {e}")

                    # 继续处理下一个音频。
                    continue

                # 每处理 500 个音频打印一次进度，防止长时间无输出。
                if (i + 1) % 500 == 0:
                    # i 从 0 开始，所以显示进度时要加 1。
                    print(f"  进度: {i+1}/{len(missing)}")

            # 缓存生成循环结束后打印完成信息。
            print(f"[预计算] 完成，生成 {len(missing)} 个缓存文件")

        # 重新遍历所有音频，只把缓存文件确实存在的样本加入最终样本列表。
        for audio_path, cache_path, label_idx in audio_files:
            # 如果缓存存在，说明该样本可以用于训练或验证。
            if os.path.exists(cache_path):
                # samples 中只保存缓存路径和标签，训练时不用再读原始音频。
                self.samples.append((cache_path, label_idx))

        # 打印最终可用样本数量和类别数。
        print(f"[数据集] 共找到 {len(self.samples)} 条音频(缓存)，{NUM_CLASSES} 个类别")

    # __len__ 告诉 DataLoader 这个数据集一共有多少个样本。
    def __len__(self):
        # 返回可用缓存样本数量。
        return len(self.samples)

    # __getitem__ 告诉 DataLoader 按索引 idx 取一个样本时该怎么做。
    def __getitem__(self, idx):
        # 从 samples 中取出缓存路径和类别标签。
        cache_path, label = self.samples[idx]

        # 从 .pt 文件加载 Mel 频谱；weights_only=True 表示只按张量/权重方式安全读取。
        mel = torch.load(cache_path, weights_only=True)

        # 返回一个训练样本：Mel 频谱张量和它的类别编号。
        return mel, label


# ────────────────────────────────────────────────
# 在线数据增强包装器：训练集使用，验证集不使用
# ────────────────────────────────────────────────


# 定义训练集包装器；它从缓存数据集中取子集，并添加随机增强。
class TrainMelDataset(Dataset):
    """包装 PrecomputedMelDataset 的训练子集，施加 Time/Freq Mask 增强并归一化。"""

    # base_dataset 是完整缓存数据集，indices 是训练集样本索引列表。
    def __init__(self, base_dataset, indices):
        # 保存完整缓存数据集引用。
        self.base = base_dataset

        # 保存训练集要使用的样本索引。
        self.indices = indices

        # 创建时间遮挡增强；会随机遮住 Mel 频谱图横轴上的一段时间。
        self.time_mask = T.TimeMasking(time_mask_param=TIME_MASK_PARAM)

        # 创建频率遮挡增强；会随机遮住 Mel 频谱图纵轴上的一段频率。
        self.freq_mask = T.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)

    # 返回训练子集长度。
    def __len__(self):
        # 长度等于训练索引列表的长度。
        return len(self.indices)

    # 取训练集中第 idx 个样本。
    def __getitem__(self, idx):
        # self.indices[idx] 先映射到完整数据集中的真实索引，再取出 Mel 和标签。
        mel, label = self.base[self.indices[idx]]

        # 如果允许时间平移增强，就进入平移逻辑。
        if MAX_TIME_SHIFT > 0:
            # 随机生成一个整数平移量，范围是 [-MAX_TIME_SHIFT, MAX_TIME_SHIFT]。
            shift = torch.randint(-MAX_TIME_SHIFT, MAX_TIME_SHIFT + 1, (1,)).item()

            # 如果 shift 不是 0，就真的平移；0 表示保持原样。
            if shift:
                # 沿最后一维时间轴滚动 Mel 图，模拟声音事件出现得早一点或晚一点。
                mel = torch.roll(mel, shifts=shift, dims=-1)

        # 对时间轴做随机遮挡，让模型不要依赖某一小段固定声音。
        mel = self.time_mask(mel)

        # 对频率轴做随机遮挡，让模型不要依赖某一小段固定频率。
        mel = self.freq_mask(mel)

        # 计算当前 Mel 图中的最小值，用于归一化。
        mel_min = mel.min()

        # 计算当前 Mel 图中的最大值，用于归一化。
        mel_max = mel.max()

        # 如果最大值大于最小值，说明这个样本不是常数图，可以安全归一化。
        if mel_max > mel_min:
            # 把数值缩放到 [0, 1]，让不同音频的数值范围更统一。
            mel = (mel - mel_min) / (mel_max - mel_min)

        # 返回增强并归一化后的 Mel 图，以及原始类别标签。
        return mel, label


# 定义验证集包装器；验证集只做归一化，不做随机增强。
class ValMelDataset(Dataset):
    """包装 PrecomputedMelDataset 的验证子集，仅归一化，不做增强。"""

    # base_dataset 是完整缓存数据集，indices 是验证集样本索引列表。
    def __init__(self, base_dataset, indices):
        # 保存完整缓存数据集引用。
        self.base = base_dataset

        # 保存验证集要使用的样本索引。
        self.indices = indices

    # 返回验证子集长度。
    def __len__(self):
        # 长度等于验证索引列表的长度。
        return len(self.indices)

    # 取验证集中第 idx 个样本。
    def __getitem__(self, idx):
        # 从完整缓存数据集中取出对应 Mel 图和标签。
        mel, label = self.base[self.indices[idx]]

        # 计算当前 Mel 图最小值。
        mel_min = mel.min()

        # 计算当前 Mel 图最大值。
        mel_max = mel.max()

        # 只有在最大值大于最小值时才归一化，避免除以 0。
        if mel_max > mel_min:
            # 验证集也缩放到 [0, 1]，保证和训练输入范围一致。
            mel = (mel - mel_min) / (mel_max - mel_min)

        # 返回验证样本和标签。
        return mel, label


# ────────────────────────────────────────────────
# DataLoader 构建：划分训练/验证，并计算类别权重
# ────────────────────────────────────────────────


# 定义分层划分函数；labels 是所有样本的类别编号列表。
def _stratified_split(labels, train_ratio: float, seed: int):
    """按类别分层划分，保持训练/验证集的类别比例稳定。"""
    # 创建独立随机数生成器，并用 seed 固定它，保证划分结果可复现。
    generator = torch.Generator().manual_seed(seed)

    # 将 labels 转成 long 类型张量，方便后面用张量操作查找类别索引。
    labels = torch.as_tensor(labels, dtype=torch.long)

    # 创建训练索引和验证索引列表。
    train_indices, val_indices = [], []

    # 对每个类别分别划分训练/验证，避免某些类别全被分到训练集或验证集。
    for label_idx in range(NUM_CLASSES):
        # 找到属于当前类别的所有样本索引。
        class_indices = torch.nonzero(labels == label_idx, as_tuple=False).flatten()

        # 对当前类别内部索引随机打乱。
        shuffled = class_indices[torch.randperm(len(class_indices), generator=generator)]

        # 根据训练比例计算当前类别应放入训练集的样本数量。
        n_train = int(len(shuffled) * train_ratio)

        # 如果当前类别样本数大于 1，就尽量保证训练集和验证集都至少有一个样本。
        if len(shuffled) > 1:
            # max(n_train, 1) 保证训练至少 1 个；min(..., len-1) 保证验证至少 1 个。
            n_train = min(max(n_train, 1), len(shuffled) - 1)

        # 把当前类别前 n_train 个索引加入训练集。
        train_indices.extend(shuffled[:n_train].tolist())

        # 把当前类别剩下的索引加入验证集。
        val_indices.extend(shuffled[n_train:].tolist())

    # 将训练索引列表转成张量，方便再次整体随机打乱。
    train_indices = torch.as_tensor(train_indices)

    # 将验证索引列表转成张量，方便再次整体随机打乱。
    val_indices = torch.as_tensor(val_indices)

    # 打乱训练索引顺序，避免 DataLoader 前期总是看到同一类。
    train_indices = train_indices[torch.randperm(len(train_indices), generator=generator)].tolist()

    # 打乱验证索引顺序，让验证遍历顺序也更均匀。
    val_indices = val_indices[torch.randperm(len(val_indices), generator=generator)].tolist()

    # 返回训练集索引和验证集索引。
    return train_indices, val_indices


# 定义类别权重计算函数；样本少的类别会得到更大的权重。
def _class_weights(labels):
    # bincount 统计每个类别在训练集中出现了多少次。
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=NUM_CLASSES).float()

    # 权重公式：总样本数 / (类别数 * 当前类别样本数)，样本越少权重越大。
    weights = counts.sum() / (NUM_CLASSES * counts.clamp_min(1.0))

    # 返回浮点权重和整数形式的类别计数。
    return weights, counts.long()


# 定义主入口函数；训练脚本调用它来得到训练和验证 DataLoader。
def get_dataloaders(data_dir: str, cache_dir: str):
    """构建训练集 / 验证集 DataLoader（使用预计算缓存）。"""
    # 先创建完整缓存数据集；如果缓存缺失，会在这里自动生成。
    base_dataset = PrecomputedMelDataset(data_dir, cache_dir)

    # 从完整数据集样本列表中提取每个样本的类别标签。
    labels = [label for _, label in base_dataset.samples]

    # 按类别分层划分训练集和验证集索引。
    train_indices, val_indices = _stratified_split(labels, TRAIN_RATIO, RANDOM_SEED)

    # 用训练索引包装成训练数据集；它会做随机增强。
    train_dataset = TrainMelDataset(base_dataset, train_indices)

    # 用验证索引包装成验证数据集；它只做归一化。
    val_dataset = ValMelDataset(base_dataset, val_indices)

    # 取出训练集对应的标签，用于计算类别权重。
    train_labels = [labels[i] for i in train_indices]

    # 根据训练集标签计算类别权重和每类数量。
    class_weights, class_counts = _class_weights(train_labels)

    # 创建训练 DataLoader；它负责把单个样本自动拼成 batch。
    train_loader = DataLoader(
        # 训练数据集。
        train_dataset,

        # 每个 batch 的样本数。
        batch_size=BATCH_SIZE,

        # 训练集需要打乱顺序，避免模型记住固定排列。
        shuffle=True,

        # 使用多少个子进程读取数据。
        num_workers=NUM_WORKERS,

        # pin_memory=True 在使用 GPU 时可以加速 CPU 到 GPU 的数据拷贝。
        pin_memory=True,

        # 丢弃最后一个不满 batch 的小批次，让每个训练 batch 大小一致。
        drop_last=True,

        # 让 DataLoader 子进程跨 epoch 保持存活，减少 Windows 反复启动进程的开销。
        persistent_workers=True if NUM_WORKERS > 0 else False,

        # 每个 worker 预取更多 batch，尽量减少 GPU 等待数据的时间。
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    # 创建验证 DataLoader；验证时不需要打乱，便于复现和排查。
    val_loader = DataLoader(
        # 验证数据集。
        val_dataset,

        # 验证 batch 大小沿用配置中的 BATCH_SIZE。
        batch_size=BATCH_SIZE,

        # 验证集不打乱，因为它不参与学习。
        shuffle=False,

        # 验证集读取也可以用多进程。
        num_workers=NUM_WORKERS,

        # 固定内存同样有利于 GPU 验证阶段的数据拷贝。
        pin_memory=True,

        # 验证阶段同样复用 worker，避免每轮验证前重新创建子进程。
        persistent_workers=True if NUM_WORKERS > 0 else False,

        # 验证阶段也提前预取 batch，让评估过程更连贯。
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    # 打印训练集和验证集样本数量。
    print(f"[DataLoader] 训练: {len(train_dataset)} | 验证: {len(val_dataset)}")

    # 打印训练集中每个类别的样本数量，用来观察类别是否均衡。
    print(f"[DataLoader] 训练集类别数: {class_counts.tolist()}")

    # 返回训练加载器、验证加载器和类别权重。
    return train_loader, val_loader, class_weights


# ────────────────────────────────────────────────
# 原版在线数据集：保留备用，但默认训练不走这条路径
# ────────────────────────────────────────────────


# 定义在线数据集；每次 __getitem__ 都从原始音频重新计算 Mel 频谱。
class EatingSoundDataset(Dataset):
    """原始在线数据集：每次从原始音频计算 Mel 频谱（可选增强）。"""

    # data_dir 是原始音频根目录，augment 决定是否使用随机增强。
    def __init__(self, data_dir: str, augment: bool = False):
        # 保存样本列表，每个元素是 (原始音频路径, 类别编号)。
        self.samples = []

        # 保存是否做增强的开关。
        self.augment = augment

        # 预先计算统一音频长度对应的采样点数量。
        self.n_samples = int(SAMPLE_RATE * DURATION)

        # 遍历所有类别文件夹。
        for label_idx, class_name in enumerate(CLASSES):
            # 拼出当前类别目录。
            class_dir = os.path.join(data_dir, class_name)

            # 如果类别目录不存在，就警告并跳过。
            if not os.path.isdir(class_dir):
                # 打印缺失目录，方便用户检查数据集路径。
                print(f"[警告] 找不到文件夹: {class_dir}")

                # 跳到下一个类别。
                continue

            # 遍历当前类别目录中的所有文件。
            for fname in os.listdir(class_dir):
                # 只接收常见音频文件格式。
                if fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    # 把音频路径和类别编号加入样本列表。
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

        # 打印在线数据集扫描到的样本数量。
        print(f"[数据集(在线)] 共找到 {len(self.samples)} 条音频，{NUM_CLASSES} 个类别")

        # 创建训练时可用的时间遮挡增强。
        self.time_mask = T.TimeMasking(time_mask_param=TIME_MASK_PARAM)

        # 创建训练时可用的频率遮挡增强。
        self.freq_mask = T.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)

    # 返回在线数据集样本数量。
    def __len__(self):
        # len(self.samples) 就是扫描到的音频文件数量。
        return len(self.samples)

    # 每次按索引读取一个原始音频，并即时转换成 Mel 频谱。
    def __getitem__(self, idx):
        # 取出当前样本的音频路径和类别标签。
        path, label = self.samples[idx]

        # 从硬盘读取音频波形和采样率。
        waveform, sr = torchaudio.load(path)

        # 执行统一的音频到 dB Mel 频谱转换。
        mel = _audio_to_mel_db(waveform, sr)

        # 如果 augment=True，就对训练样本做随机遮挡增强。
        if self.augment:
            # 随机遮挡一段时间区域。
            mel = self.time_mask(mel)

            # 随机遮挡一段频率区域。
            mel = self.freq_mask(mel)

        # 计算 Mel 图最小值。
        mel_min = mel.min()

        # 计算 Mel 图最大值。
        mel_max = mel.max()

        # 避免最大值等于最小值时除以 0。
        if mel_max > mel_min:
            # 把 Mel 图缩放到 [0, 1]。
            mel = (mel - mel_min) / (mel_max - mel_min)

        # 返回处理后的 Mel 频谱图和类别标签。
        return mel, label
