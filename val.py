import torch
import torch.nn as nn
import torchvision.transforms as transforms
from license_plate_dataset import LicensePlateDataset, LicensePlateVocab
from license_plate_model import LicensePlateModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.font_manager as fm
from tqdm import tqdm
import os
import argparse
    
img_height = 224
img_width = 224

# 最大序列长度
max_length = 16 
num_epochs = 100


# 设置中文字体
font_path = "C:/Windows/Fonts/simhei.ttf"  # 可以选择其他中文字体
prop = fm.FontProperties(fname=font_path)

# 词汇表
vocab_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']
vocab = LicensePlateVocab(vocab_list)


def generate_license_plate(model, image, vocab, device, max_length=16):
    with torch.no_grad():
        memory = model.encode_image(image)
        
        expression = ''
        generated_tokens = vocab.text_to_sequence(expression, max_length=max_length, pad_to_max_length=False, add_eos=False, add_bos=True)
        while True:
            tgt_input = torch.tensor(generated_tokens).unsqueeze(0).to(device)
            output = model.decode_text(memory, tgt_input)
            next_token = output.argmax(dim=-1)[:, -1].item()
            if next_token == vocab.eos_idx or len(expression) == max_length:
                break
            expression += vocab.sequence_to_text([next_token])
            generated_tokens = vocab.text_to_sequence(expression, max_length=max_length, pad_to_max_length=False, add_eos=False, add_bos=True)
    
    return expression


def generate_license_plate_once(model, image, vocab, device, max_length=16):
    with torch.no_grad():
        memory = model.encode_image(image)
        expression = ''
        generated_tokens = vocab.text_to_sequence(expression, max_length=max_length, pad_to_max_length=True, add_eos=False, add_bos=True)
        tgt_input = torch.tensor(generated_tokens).unsqueeze(0).to(device)
        output = model.decode_text(memory, tgt_input)
        next_token = output.argmax(dim=-1).cpu().numpy()[0]
        # 裁剪掉EOS之后的数据
        if vocab.eos_idx in next_token:
            eos_index = list(next_token).index(vocab.eos_idx)
            next_token = next_token[:eos_index]
        expression = vocab.sequence_to_text(next_token)
    return expression


# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    exact_match_correct = 0
    total = 0
    images_count = 0
    batch_idx = 0
    val_loader_tqdm = tqdm(val_loader, desc='Validation', unit='batch', leave=False)
    
    with torch.no_grad():
        for images, labels in val_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, labels[:, :-1])
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, labels[:, 1:])
            val_loss += loss.item() * images.size(0)
            images_count += images.size(0)

            _, predicted = torch.max(outputs, 1)
            
            # 创建一个掩码，标识哪些位置不是 vocab.pad_idx
            non_pad_mask = (labels[:, 1:] != vocab.pad_idx)
            pad_mask = (labels[:, 1:] == vocab.pad_idx)
            predicted[pad_mask] = vocab.pad_idx

            # 使用掩码来计算正确的预测
            correct += (predicted[non_pad_mask] == labels[:, 1:][non_pad_mask]).sum().item()
            total += non_pad_mask.sum().item()

            # 计算每个序列的完全匹配情况
            exact_match_correct += ((predicted == labels[:, 1:]) | ~non_pad_mask).all(dim=1).sum().item()
            
            val_loader_tqdm.set_postfix(loss=f"{val_loss/images_count:.5f}")

            # Plotting the results
            fig, axes = plt.subplots(10, 10, figsize=(15, 15))
            axes = axes.flatten()

            for i, (image, label, pred) in enumerate(zip(images, labels, predicted)):
                image_np = image.cpu().numpy().transpose(1, 2, 0)
                real_plate = vocab.sequence_to_text(label.cpu().numpy()[1:])  # Skip the start token
                predicted_plate = vocab.sequence_to_text(pred.cpu().numpy())

                axes[i].imshow(image_np)
                if real_plate != predicted_plate:
                    axes[i].set_title(f'标注: {real_plate}\n预测: {predicted_plate}', fontproperties=prop, color='red')
                else:
                    axes[i].set_title(f'标注: {real_plate}\n预测: {predicted_plate}', fontproperties=prop)
                axes[i].axis('off')

            plt.tight_layout()
            os.makedirs('output_images', exist_ok=True)
            plt.savefig(f'output_images/generated_plates_{batch_idx:05d}.png')
            batch_idx += 1
            plt.close()

    val_loss = val_loss / images_count
    accuracy = correct / total
    exact_match_accuracy = exact_match_correct / images_count
    return val_loss, accuracy, exact_match_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate a license plate recognition model")
    parser.add_argument('--val_folder', type=str, default=r'./datasets/test', help='Path to the validation dataset folder')
    parser.add_argument('--checkpoint_path', type=str, default=r'./checkpoints/last_model.pth', help='Path to the model checkpoint file')
    args = parser.parse_args()
    
    # 设置数据变换
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()
    ])

    # 创建数据集和数据加载器
    val_dataset = LicensePlateDataset(args.val_folder, vocab, max_length, transform)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=4)

    data_iter = iter(val_loader)
    images, labels = next(data_iter)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = LicensePlateModel(pad_idx=vocab.pad_idx, vocab_size=len(vocab.vocab_list), max_length=max_length)
    model = LicensePlateModel(pad_idx=vocab.pad_idx, d_model=64, nhead_encoder=4, nhead_decoder=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, vocab_size=len(vocab.vocab_list), max_length=max_length)
    model.load_state_dict(torch.load(args.checkpoint_path)['model_state_dict'])
    model.to(device)
    
    val_loss, accuracy, exact_match_accuracy = validate(model, val_loader, nn.CrossEntropyLoss(ignore_index=vocab.pad_idx), device)
    print(f"Validation Loss: {val_loss:.5f}, Accuracy: {accuracy:.5f}, Exact Match Accuracy: {exact_match_accuracy:.5f}")
    