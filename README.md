# 中文车牌识别模型（基于Transformer架构）

## 概述

本项目实现了一个基于Transformer架构的中文车牌识别系统。该模型由两个主要部分组成：图像编码器和文本解码器，分别处理车牌图像和预测的车牌文本序列。这些组件集成在一个综合的模型中。

![车牌识别结果](https://github.com/sosopop/chinese-lpr-transformer/blob/main/assets/generated_plates_00000.png)

## 模型特点

- **高准确率**：在普通车牌数据集和高难度数据集上表现优异。
- **鲁棒性**：在车牌变形和清晰度不高的情况下依然保持高识别率。
- **多种车牌支持**：支持黄牌、蓝牌、绿牌车牌。
- **灵活性**：支持非固定长度和布局的车牌号。
- **通用性**：该架构可用于通用的OCR识别任务，适用于各种文本识别场景。

## 模型架构

![模型架构图](https://github.com/sosopop/chinese-lpr-transformer/blob/main/assets/model_diagram.png)

### 图像编码器（ImageEncoder）
- **骨干网络（Backbone）**: 使用 MobileNetV3_Small 提取车牌图像的高级特征。
- **卷积层（Conv Layer）**: 将特征图通道数减少到模型的嵌入维度（128）。
- **二维位置编码（PositionalEncoding2D）**: 为特征图添加空间信息。
- **Transformer 编码器（Transformer Encoder）**: 处理图像特征以捕捉长距离依赖关系。

### 文本解码器（TextDecoder）
- **嵌入层（Embedding Layer）**: 将目标序列转换为嵌入表示。
- **一维位置编码（PositionalEncoding1D）**: 为序列添加位置信息。
- **Transformer 解码器（Transformer Decoder）**: 通过与图像特征和目标序列交互生成预测序列。
- **全连接层（Fully Connected Layer）**: 将解码器输出转换为词汇表中每个字符的概率分布。

## 数据集

本项目使用 CCPD 和 CRPD 两个数据集的数据整理而成。您可以通过以下链接下载这些数据集：

- [数据集下载](https://github.com/)

## 快速开始

### 先决条件

- Python 3.x
- PyTorch
- ONNX
- `requirements.txt` 中列出的其他依赖项

### 安装

```bash
git clone https://github.com/sosopop/chinese-lpr-transformer.git
cd chinese-lpr-transformer
pip install -r requirements.txt
```

### 使用


### 训练

使用提供的训练脚本训练模型：

```bash
python train.py --data_path /path/to/your/data --epochs 50 --batch_size 32
```

### 评估

使用评估脚本在测试数据集上评估模型：

```bash
python val.py --data_path /path/to/your/data --batch_size 32
```

### 导出为 ONNX

将训练好的模型导出为 ONNX 格式以便部署：

```bash
python export.py --model /path/model.pth --output /model_path/
```

## Android 演示

在 `android_demo` 目录中包含了一个 Android 演示。按照以下步骤设置并运行演示。

## 准确率对比

下表展示了本模型与 [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch) 在普通车牌数据集和高难度数据集上的准确率对比：

| 数据集类型 | LPRNet_Pytorch 准确率 | 本模型准确率 |
|------------|----------------------|------------|
| 普通车牌数据集 | 0.898                | 0.987      |
| 高难度数据集   | 0.064                | 0.8447     |


## 联系方式

如有问题或意见，请打开问题或联系项目维护者：[12178761@qq.com]。欢迎添加我的微信：**mengchao1102**。

---

请点击Star来支持这个项目，谢谢！
