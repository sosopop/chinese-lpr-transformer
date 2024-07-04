import numpy as np
import onnxruntime
import argparse
from PIL import Image
from torchvision import transforms
from license_plate_dataset import LicensePlateVocab
import os

# 定义词汇表
max_length = 16
vocab_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']
vocab = LicensePlateVocab(vocab_list)

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 定义推理函数
def generate_license_plate_with_memory_onnx(encoder_ort_session, decoder_ort_session, image, vocab, max_length=16):
    memory = encoder_ort_session.run(None, {'image': image})
    memory = memory[0]
    
    expression = ''
    generated_tokens = vocab.text_to_sequence(expression, max_length=max_length, pad_to_max_length=True, add_eos=False, add_bos=True)
    while True:
        tgt_input = np.array(generated_tokens).reshape(1, -1).astype(np.int64)
        ort_inputs = {'memory': memory, 'tgt': tgt_input}
        ort_outs = decoder_ort_session.run(None, ort_inputs)
        output = ort_outs[0]
        next_token = np.argmax(output, axis=-1)[:, len(expression)].item()
        expression += vocab.sequence_to_text([next_token])
        generated_tokens = vocab.text_to_sequence(expression, max_length=max_length, pad_to_max_length=True, add_eos=False, add_bos=True)
        if next_token == vocab.eos_idx:
            break
        if len(expression) >= max_length:
            print("\nError: expression length exceeds max_length")
            break
    
    return expression

def main(image_path, encoder_path, decoder_path):
    # 加载 ONNX 模型
    encoder_ort_session = onnxruntime.InferenceSession(encoder_path)
    decoder_ort_session = onnxruntime.InferenceSession(decoder_path)

    # 读取图片并进行预处理
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).numpy()
    
    # 进行推理
    expression = generate_license_plate_with_memory_onnx(encoder_ort_session, decoder_ort_session, image_tensor, vocab, max_length)
    
    # 打印推理结果
    print(f"Predicted: {expression}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX model inference for a single license plate image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--encoder_path", type=str, default="./output_models/image_encoder.onnx", help="Path to the ONNX encoder model file")
    parser.add_argument("--decoder_path", type=str, default="./output_models/text_decoder.onnx", help="Path to the ONNX decoder model file")

    args = parser.parse_args()

    main(args.image_path, args.encoder_path, args.decoder_path)
