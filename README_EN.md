# Chinese License Plate Recognition Model (Based on Transformer End-to-End Architecture)

[简体中文](./README.md) | [English](./README_EN.md)

## Overview

This project implements a highly accurate Chinese license plate recognition model based on the Transformer architecture. Leveraging advanced deep learning techniques, the model demonstrates outstanding performance in various complex scenarios for license plate recognition tasks.

We utilize the lightweight yet powerful MobileNetV3 as the backbone network, combined with the strong sequential modeling capabilities of the Transformer architecture, ensuring high efficiency and accuracy in processing license plate images. Whether dealing with standard license plate datasets or challenging ones, our model consistently provides near-perfect recognition results.

Moreover, our model exhibits high robustness, capable of handling challenges such as plate deformations and blurriness commonly found in real-world scenarios. It supports various types of plates including yellow, blue, and green plates, and adapts well to non-fixed lengths and layouts of license plate numbers. The versatility of this model extends beyond license plate recognition to various OCR tasks such as card number recognition and captcha recognition, significantly enhancing its practical utility and application value.

![License Plate Recognition Result](https://github.com/sosopop/chinese-lpr-transformer/blob/main/assets/generated_plates_00000.png)

## Contact Information

For questions or feedback, please open an issue or contact the project maintainer. You can also add me on WeChat: **mengchao1102** for technical discussions. If you find this project helpful, please consider giving it a star to show your support. Thank you.

## Model Features

- **High Accuracy**: Excellent performance on both standard and challenging license plate datasets.
- **Robustness**: Maintains high recognition rates even with plate deformations and low image clarity.
- **Multiple Plate Support**: Recognizes yellow, blue, green plates, etc.
- **Flexibility**: Handles non-fixed lengths and layouts of license plate numbers.
- **Versatility**: Suitable for general OCR tasks, applicable to various text recognition scenarios such as card numbers and captchas.

## Model Architecture

![Model Architecture Diagram](https://github.com/sosopop/chinese-lpr-transformer/blob/main/assets/model_diagram.png)

### Image Encoder (ImageEncoder)

- **Backbone Network**: Uses MobileNetV3_Small to extract advanced features from license plate images.
- **Convolutional Layers (Conv Layer)**: Reduces the number of channels in feature maps to the embedding dimension (64).
- **2D Positional Encoding (PositionalEncoding2D)**: Adds spatial information to the feature maps.
- **Transformer Encoder**: Processes image features to capture long-range dependencies.

### Text Decoder (TextDecoder)

- **Embedding Layer**: Converts target sequences into embedding representations.
- **1D Positional Encoding (PositionalEncoding1D)**: Adds positional information to the sequence.
- **Transformer Decoder**: Generates predicted sequences by interacting with image features and target sequences.
- **Fully Connected Layer**: Converts decoder outputs into probability distributions for each character in the vocabulary.

## Accuracy Comparison

The table below compares the accuracy of this model with [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch) on standard and challenging license plate datasets:

| Dataset Type      | LPRNet_Pytorch Accuracy | This Model Accuracy |
|-------------------|-------------------------|---------------------|
| Standard Dataset  | 89.8%                   | 99.3%               |
| Challenging Dataset | 6.4%                   | 85.7%               |

## Datasets

This project compiles data from CCPD and CRPD datasets. You can download these datasets from the following links:

Dataset Download Links:

[Baidu Pan](https://pan.baidu.com/s/18YfphNe0yQeJrISwtGD_wg?pwd=1d7a) Password: `1d7a`

After downloading, unzip the data into the `datasets` directory of the project.

## Model Files

This project provides pre-trained model files including:

- **last_model.pth**: Pre-trained model weights file.

Model Download Links:

[Baidu Pan](https://pan.baidu.com/s/11WVX91QVwY_0qGdy3mxfBA?pwd=yvzp) Password: `yvzp`

After downloading, copy the `last_model.pth` file into the `checkpoints` directory of the project. You can use the `onnx_export.py` script to export an ONNX model file.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- ONNX

### Installation

```bash
git clone https://github.com/sosopop/chinese-lpr-transformer.git
```

### Training

Train the model using the provided training script:

```bash
python train.py
```

### Evaluation

Evaluate the model on the test dataset using the evaluation script:

```bash
python val.py
```

### Export to ONNX

Export the trained model to ONNX format for deployment:

```bash
python onnx_export.py --weight ./checkpoints/last_model.pth --output ./output_models
```

The exported ONNX files include:

- **image_encoder.onnx**: Model containing the image encoder responsible for encoding license plate images into high-level feature representations.
- **text_decoder.onnx**: Model containing the text decoder responsible for decoding the feature representations generated by the image encoder into license plate character sequences.
- **complete_model.onnx**: Model combining both the image encoder and text decoder to perform end-to-end license plate recognition.

These ONNX files can be used on different platforms and devices such as embedded devices, mobile devices, or servers, facilitating model deployment and inference.

### ONNX Inference Testing

```bash
python onnx_inference.py --image_path datasets/test/川A023Y9-00017091.jpg --encoder_path ./output_models/image_encoder.onnx --decoder_path ./output_models/text_decoder.onnx
```

## Android Demo

The `android_demo` directory contains an Android demo. Follow these steps to set up and run the demo.

### Setup

1. Open Android Studio and import the `android_demo` project.
2. Copy `image_encoder.onnx` and `text_decoder.onnx` files into the `android_demo/LicensePlateRecognition/app/src/main/assets` directory.
3. Build and run the project on your Android device or emulator.

## Reference Projects

- [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)
- [A PyTorch implementation of MobileNetV3](https://github.com/xiaolai-sqlai/mobilenetv3)
