# model.py
# ============================================================
# CNN 模型：把 Mel 频谱图当作单通道图片来做 20 类声音分类
# ============================================================
#
# 给初学者的理解方式：
# Mel 频谱图可以看成一张“声音图片”，横轴是时间，纵轴是频率，
# 图片上越亮的位置表示对应时间和频率处声音能量越强。
# CNN 擅长从图片里找局部模式，所以这里用卷积网络寻找“咀嚼、脆裂、液体”等声音纹理。

# 导入 PyTorch 主库；这里主要用于类型和张量相关功能。
import torch

# 导入神经网络模块；卷积层、归一化层、激活函数、全连接层都来自 nn。
import torch.nn as nn

# 从配置文件导入默认 Dropout 概率、默认模型结构名称和类别数量。
from config import DROPOUT, MODEL_ARCH, NUM_CLASSES


# 定义基础卷积块；它是 cnn_v1 使用的最小特征提取单元。
class ConvBlock(nn.Module):
    # in_ch 是输入通道数，out_ch 是输出通道数，pool 决定是否做下采样。
    def __init__(self, in_ch, out_ch, pool=True):
        # 调用父类初始化；所有 nn.Module 子类都应该先执行这句。
        super().__init__()

        # layers 用列表暂存一组神经网络层，最后会打包成 nn.Sequential。
        layers = [
            # 3x3 卷积提取局部时间-频率模式；padding=1 保持宽高不变。
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),

            # BatchNorm2d 对卷积输出做标准化，让训练更稳定、更容易收敛。
            nn.BatchNorm2d(out_ch),

            # ReLU 激活函数引入非线性，让模型能学习复杂声音模式。
            nn.ReLU(inplace=True),
        ]

        # 如果 pool=True，就在卷积块末尾加入最大池化，把特征图宽高减半。
        if pool:
            # MaxPool2d(2, 2) 每 2x2 区域取最大值，保留最强特征并降低计算量。
            layers.append(nn.MaxPool2d(2, 2))

        # 把上面列表里的层按顺序组合起来，形成一个可直接调用的模块。
        self.block = nn.Sequential(*layers)

    # forward 定义数据经过这个模块时的计算流程。
    def forward(self, x):
        # 将输入 x 依次送过卷积、BN、激活和可选池化。
        return self.block(x)


# 定义第一版进食声音 CNN；结构较浅，主要用于兼容旧模型或做基线对比。
class EatingSoundCNN(nn.Module):
    """
    输入形状: (B, 1, 128, T)
    B 是 batch size，1 表示单通道，128 是 Mel 频带数，T 是时间帧数。
    输出形状: (B, NUM_CLASSES)
    每一行输出是一个样本对 20 个类别的原始分数 logits。
    """

    # num_classes 是类别数，dropout 是分类器里的随机失活概率。
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        # 初始化 nn.Module 父类。
        super().__init__()

        # features 是特征提取器，负责从 Mel 频谱图中逐层提取声音纹理。
        self.features = nn.Sequential(
            # 第 1 个卷积块：输入 1 通道，输出 32 通道，频率高度 128 变成 64。
            ConvBlock(1, 32, pool=True),

            # 第 2 个卷积块：通道 32 -> 64，频率高度 64 变成 32。
            ConvBlock(32, 64, pool=True),

            # 第 3 个卷积块：通道 64 -> 128，频率高度 32 变成 16。
            ConvBlock(64, 128, pool=True),

            # 第 4 个卷积块：通道 128 -> 256，频率高度 16 变成 8。
            ConvBlock(128, 256, pool=True),
        )

        # 全局平均池化把任意宽高的特征图压缩成固定的 1x1。
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # classifier 是分类头，负责把 256 维特征转换成 20 类分数。
        self.classifier = nn.Sequential(
            # Flatten 把 (B, 256, 1, 1) 拉平成 (B, 256)。
            nn.Flatten(),

            # Dropout 随机屏蔽一部分特征，降低模型死记训练集的风险。
            nn.Dropout(dropout),

            # 全连接层把 256 维特征压到 128 维，进一步组合高级特征。
            nn.Linear(256, 128),

            # ReLU 增加非线性表达能力。
            nn.ReLU(inplace=True),

            # 第二个 Dropout 稍弱一些，继续帮助模型泛化。
            nn.Dropout(dropout / 2),

            # 最后一层输出 num_classes 个 logits，对应每个类别的未归一化分数。
            nn.Linear(128, num_classes),
        )

    # forward 定义一次前向传播从输入到输出的完整路径。
    def forward(self, x):
        # 先通过卷积特征提取器，得到多通道二维特征图。
        x = self.features(x)

        # 再用全局平均池化压缩空间维度，保留每个通道的整体响应。
        x = self.gap(x)

        # 最后进入分类头，输出每个类别的 logits。
        x = self.classifier(x)

        # 返回 logits；训练时 CrossEntropyLoss 会直接使用它。
        return x


# 定义双卷积块；cnn_v2 每个阶段用两层卷积，所以表达能力更强。
class DoubleConvBlock(nn.Module):
    # in_ch 是输入通道，out_ch 是输出通道，pool 控制池化，dropout 控制二维 Dropout。
    def __init__(self, in_ch, out_ch, pool=True, dropout=0.0):
        # 初始化父类。
        super().__init__()

        # 用列表保存本阶段的所有网络层。
        layers = [
            # 第一层 3x3 卷积，负责把输入通道转换为 out_ch。
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),

            # 第一层 BatchNorm，让每个通道的数值分布更稳定。
            nn.BatchNorm2d(out_ch),

            # SiLU 激活函数比 ReLU 更平滑，常能带来更稳定的训练效果。
            nn.SiLU(inplace=True),

            # 第二层 3x3 卷积，在同一通道数下继续提取更复杂的局部特征。
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),

            # 第二层 BatchNorm，继续稳定特征分布。
            nn.BatchNorm2d(out_ch),

            # 第二个 SiLU 激活，增强模型表达能力。
            nn.SiLU(inplace=True),
        ]

        # 如果 dropout 大于 0，就添加 Dropout2d，随机丢弃整张特征通道。
        if dropout > 0:
            # Dropout2d 常用于卷积特征图，能防止模型过度依赖某几个通道。
            layers.append(nn.Dropout2d(dropout))

        # 如果 pool=True，就做 2x2 最大池化，下采样宽高。
        if pool:
            # 池化会降低计算量，同时让后续层看到更大范围的上下文。
            layers.append(nn.MaxPool2d(2, 2))

        # 把层列表打包成顺序模块。
        self.block = nn.Sequential(*layers)

    # 定义双卷积块的前向传播。
    def forward(self, x):
        # 直接按顺序执行 block 中的所有层。
        return self.block(x)


# 定义第二版进食声音 CNN；当前配置默认使用这个更强的模型。
class EatingSoundCNNV2(nn.Module):
    """
    更强一点的 CNN：每个 stage 使用两层卷积，容量约为 v1 的 3 倍。
    输入形状: (B, 1, 128, T)
    输出形状: (B, NUM_CLASSES)
    """

    # num_classes 控制输出类别数，dropout 控制分类器里的正则化强度。
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        # 初始化父类。
        super().__init__()

        # features 是 cnn_v2 的卷积特征提取器。
        self.features = nn.Sequential(
            # Stage 1：从单通道 Mel 图提取 32 个基础声音纹理通道。
            DoubleConvBlock(1, 32, pool=True, dropout=0.05),

            # Stage 2：通道数提高到 64，开始组合更复杂的时间-频率局部特征。
            DoubleConvBlock(32, 64, pool=True, dropout=0.05),

            # Stage 3：通道数提高到 128，并使用更强 Dropout2d 防止过拟合。
            DoubleConvBlock(64, 128, pool=True, dropout=0.10),

            # Stage 4：通道数提高到 256，形成最终高级声音表示。
            DoubleConvBlock(128, 256, pool=True, dropout=0.10),
        )

        # 自适应平均池化把每个通道压成 1 个数字，得到固定长度特征。
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器把 256 通道的全局特征变成类别分数。
        self.classifier = nn.Sequential(
            # 将 (B, 256, 1, 1) 转换成 (B, 256)。
            nn.Flatten(),

            # Dropout 降低过拟合，让模型不要只记住训练样本。
            nn.Dropout(dropout),

            # 全连接层 256 -> 256，用于重新组合卷积提取到的高级特征。
            nn.Linear(256, 256),

            # SiLU 激活使分类器具备非线性表达能力。
            nn.SiLU(inplace=True),

            # 再做一次较弱 Dropout，让分类头更稳。
            nn.Dropout(dropout / 2),

            # 输出层生成 num_classes 个 logits，每个类别一个分数。
            nn.Linear(256, num_classes),
        )

    # 定义 cnn_v2 的前向传播。
    def forward(self, x):
        # 先用卷积层提取声音图像特征。
        x = self.features(x)

        # 用全局平均池化把二维特征图压成固定大小。
        x = self.gap(x)

        # 送入分类器得到 20 类 logits。
        x = self.classifier(x)

        # 返回模型输出；后续训练函数会计算损失和准确率。
        return x


# 定义模型工厂函数；训练和预测都通过它按名称创建模型。
def build_model(arch: str = MODEL_ARCH, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT) -> nn.Module:
    # 将 arch 转成小写，并在传入 None 或空字符串时默认使用 cnn_v1。
    arch = (arch or "cnn_v1").lower()

    # 如果选择 cnn_v1，就返回较浅的 EatingSoundCNN。
    if arch == "cnn_v1":
        # 创建并返回 v1 模型实例。
        return EatingSoundCNN(num_classes=num_classes, dropout=dropout)

    # 如果选择 cnn_v2，就返回当前推荐的 EatingSoundCNNV2。
    if arch == "cnn_v2":
        # 创建并返回 v2 模型实例。
        return EatingSoundCNNV2(num_classes=num_classes, dropout=dropout)

    # 如果传入未知模型名，就主动报错，避免悄悄用错结构。
    raise ValueError(f"未知模型结构: {arch}")


# 定义参数统计函数；用于训练开始时打印模型大小。
def count_params(model: nn.Module) -> str:
    # 统计模型中所有参数的元素个数，不管是否参与训练。
    total = sum(p.numel() for p in model.parameters())

    # 统计 requires_grad=True 的参数数量，也就是会被优化器更新的参数。
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 返回格式化字符串，千分位显示更方便阅读。
    return f"总参数: {total:,}  |  可训练: {trainable:,}"
