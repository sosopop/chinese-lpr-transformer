# 中文车牌识别模型（基于Transformer架构）

## 概述

本项目实现了一个基于Transformer架构的极高识别率的中文车牌识别模型。该模型在设计和实现过程中充分利用了先进的深度学习技术，使其在各种复杂场景下的车牌识别任务中表现出色。

我们采用了轻量级但强大的MobileNetV3作为骨干网络，结合Transformer架构的强大序列建模能力，确保了模型在处理车牌图像时的高效性和准确性。无论是普通的车牌数据集还是具有挑战性的高难度数据集，我们的模型都能提供接近完美的识别结果。

此外，我们的模型具有极高的鲁棒性，能够应对车牌变形、模糊等各种现实世界中的复杂情况。同时，它支持多种类型的车牌，包括黄牌、蓝牌、绿牌等，并且对非固定长度和布局的车牌号同样适用。该模型的通用性使其不仅适用于车牌识别，还可以扩展应用于各种OCR任务，如卡号识别、验证码识别等，极大地提升了其应用价值和实用性。

![车牌识别结果](https://github.com/sosopop/chinese-lpr-transformer/blob/main/assets/generated_plates_00000.png)

## 联系方式

如有问题或意见，请打开问题或联系项目维护者。欢迎添加我的微信：**mengchao1102**，进行技术交流。
如果您感觉这个项目对您有帮助，请点击Star来支持这个项目，谢谢。

## 模型特点

- **高准确率**：在普通车牌数据集和高难度数据集上表现优异。
- **鲁棒性**：在车牌变形和清晰度不高的情况下依然保持高识别率。
- **多种车牌支持**：支持黄牌、蓝牌、绿牌车牌。
- **灵活性**：支持非固定长度和布局的车牌号。
- **通用性**：该架构可用于通用的OCR识别任务，适用于各种文本识别场景，如各种卡号、验证码等。

## 模型架构

![模型架构图](https://github.com/sosopop/chinese-lpr-transformer/blob/main/assets/model_diagram.png)

### 图像编码器（ImageEncoder）
- **骨干网络（Backbone）**: 使用 MobileNetV3_Small 提取车牌图像的高级特征。
- **卷积层（Conv Layer）**: 将特征图通道数减少到模型的嵌入维度（64）。
- **二维位置编码（PositionalEncoding2D）**: 为特征图添加空间信息。
- **Transformer 编码器（Transformer Encoder）**: 处理图像特征以捕捉长距离依赖关系。

### 文本解码器（TextDecoder）
- **嵌入层（Embedding Layer）**: 将目标序列转换为嵌入表示。
- **一维位置编码（PositionalEncoding1D）**: 为序列添加位置信息。
- **Transformer 解码器（Transformer Decoder）**: 通过与图像特征和目标序列交互生成预测序列。
- **全连接层（Fully Connected Layer）**: 将解码器输出转换为词汇表中每个字符的概率分布。

## 数据集

本项目使用 CCPD 和 CRPD 两个数据集的数据整理而成。您可以通过以下链接下载这些数据集：

数据集下载地址：

[百度网盘](https://pan.baidu.com/s/18YfphNe0yQeJrISwtGD_wg?pwd=1d7a) 提取码：`1d7a`

下载完毕后解压到项目datasets目录下。

## 模型文件

本项目提供了训练好的模型文件，包括：

- **last_model.pth**: 训练好的模型权重文件。

模型下载地址：

[百度网盘](https://pan.baidu.com/s/11WVX91QVwY_0qGdy3mxfBA?pwd=yvzp) 提取码：`yvzp`

下载完毕后将last_model.pth文件复制到项目checkpoints目录下，可使用onnx_export.py脚本导出ONNX模型文件。

## 快速开始

### 先决条件

- Python 3.x
- PyTorch
- ONNX

### 安装

```bash
git clone https://github.com/sosopop/chinese-lpr-transformer.git
```

### 训练

使用提供的训练脚本训练模型：

```bash
python train.py
```

### 评估

使用评估脚本在测试数据集上评估模型：

```bash
python val.py
```

### 导出为 ONNX

将训练好的模型导出为 ONNX 格式以便部署：

```bash
python onnx_export.py --weight ./checkpoints/last_model.pth --output ./output_models
```

导出的ONNX文件说明：

- **image_encoder.onnx**: 包含图像编码器的模型。该模型负责将输入的车牌图像编码为高层次的特征表示。
- **text_decoder.onnx**: 包含文本解码器的模型。该模型负责将图像编码器生成的特征表示解码为车牌字符序列。
- **complete_model.onnx**: 包含完整的车牌识别模型。该模型结合了图像编码器和文本解码器，实现端到端的车牌识别功能。

这些ONNX文件可以在不同的平台和设备上使用，例如嵌入式设备、移动设备或服务器，方便模型的部署和推理。

### ONNX 推理测试

```bash
python onnx_inference.py --image_path datasets/test/川A023Y9-00017091.jpg --encoder_path ./output_models/image_encoder.onnx --decoder_path ./output_models/text_decoder.onnx
```

## Android 演示

在 `android_demo` 目录中包含了一个 Android 演示。按照以下步骤设置并运行演示。

### 设置

1. 打开 Android Studio 并导入 `android_demo` 项目。
2. 将 `image_encoder.onnx` 和 `text_decoder.onnx` 文件复制到 `android_demo/LicensePlateRecognition/app/src/main/assets` 目录中。
3. 在你的 Android 设备或模拟器上构建并运行项目。


## 准确率对比

下表展示了本模型与 [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch) 在普通车牌数据集和高难度数据集上的准确率对比：

| 数据集类型 | LPRNet_Pytorch 准确率 | 本模型准确率 |
|------------|----------------------|------------|
| 普通车牌数据集 | 0.898                | 0.987      |
| 高难度数据集   | 0.064                | 0.8447     |

## 参考项目

- [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)
- [A PyTorch implementation of MobileNetV3](https://github.com/xiaolai-sqlai/mobilenetv3)