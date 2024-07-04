import torch
import os
import argparse
from license_plate_model import LicensePlateModel, ImageEncoder, TextDecoder
from license_plate_dataset import LicensePlateVocab

def export_model(weight_path, output_dir):
    # 设置设备为 CPU
    device = torch.device("cpu")

    # 定义词汇表
    max_length = 16 
    vocab_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']
    vocab = LicensePlateVocab(vocab_list)

    # 加载完整模型
    model = LicensePlateModel(pad_idx=vocab.pad_idx, d_model=64, nhead_encoder=4, nhead_decoder=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, vocab_size=len(vocab.vocab_list), max_length=max_length)
    checkpoint = torch.load(weight_path, map_location=device)

    print('Loading model from checkpoint')
    print('Epoch:', checkpoint['epoch'])
    print('Train Loss:', checkpoint['loss'])
    print('Best val acc:', checkpoint['best_acc'])
    print('Best val loss:', checkpoint['best_loss'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 提取编码器和解码器
    image_encoder = model.image_encoder
    text_decoder = model.text_decoder

    # 示例输入
    image = torch.rand(1, 3, 224, 224).to(device)
    tgt = torch.ones(1, max_length).long().to(device)  # tgt 作为示例输入
    memory = image_encoder(image)  # 生成 memory 作为解码器的输入

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 导出图像编码模型为 ONNX 格式
    image_encoder_onnx_path = os.path.join(output_dir, "image_encoder.onnx")

    # 如果文件已经存在，则删除
    if os.path.exists(image_encoder_onnx_path):
        os.remove(image_encoder_onnx_path)

    torch.onnx.export(
        image_encoder, 
        image, 
        image_encoder_onnx_path, 
        input_names=['image'], 
        output_names=['memory'],
        dynamic_axes={'image': {0: 'batch_size'}, 'memory': {0: 'batch_size'}},
        opset_version=14
    )

    print(f"Image encoder model successfully exported to {image_encoder_onnx_path}")

    # 导出文字解码模型为 ONNX 格式
    text_decoder_onnx_path = os.path.join(output_dir, "text_decoder.onnx")

    # 如果文件已经存在，则删除
    if os.path.exists(text_decoder_onnx_path):
        os.remove(text_decoder_onnx_path)

    torch.onnx.export(
        text_decoder, 
        (memory, tgt), 
        text_decoder_onnx_path, 
        input_names=['memory', 'tgt'], 
        output_names=['output'],
        dynamic_axes={'memory': {0: 'batch_size'}, 'tgt': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )

    print(f"Text decoder model successfully exported to {text_decoder_onnx_path}")

    # 导出完整模型为 ONNX 格式
    complete_model_onnx_path = os.path.join(output_dir, "complete_model.onnx")

    # 如果文件已经存在，则删除
    if os.path.exists(complete_model_onnx_path):
        os.remove(complete_model_onnx_path)

    torch.onnx.export(
        model, 
        (image, tgt), 
        complete_model_onnx_path, 
        input_names=['image', 'tgt'], 
        output_names=['output'],
        dynamic_axes={'image': {0: 'batch_size'}, 'tgt': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )

    print(f"Complete model successfully exported to {complete_model_onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export license plate recognition model to ONNX format")
    parser.add_argument("--weight", type=str, default="./checkpoints/last_model.pth", help="Path to the model weight file")
    parser.add_argument("--output", type=str, default="./output_models", help="Path to the output directory")

    args = parser.parse_args()
    
    print(f"Exporting model from {args.weight} to {args.output}")

    export_model(args.weight, args.output)
