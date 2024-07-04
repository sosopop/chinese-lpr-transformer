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
import shutil
import os
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

# 定义默认参数
DEFAULT_TRAIN_FOLDER = r'./datasets/train'
DEFAULT_VAL_FOLDER = r'./datasets/val'
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_NUM_EPOCHS = 200
DEFAULT_CHECKPOINT_PATH = 'last_model.pth'

img_size = 224

# 最大序列长度
max_length = 16 
num_epochs = 200

# 设置中文字体
font_path = "C:/Windows/Fonts/simhei.ttf"  # 可以选择其他中文字体
prop = fm.FontProperties(fname=font_path)

checkpoints_folder = 'checkpoints'


# 词汇表
vocab_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']
vocab = LicensePlateVocab(vocab_list)

def get_train_transform(default_transform=True):
    if default_transform:
        chosen_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        print(f'Using default transform')
    else:
        chosen_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.05)),
            transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.001, 0.005), ratio=(0.5, 2), value=(1.0, 0.0, 0.0))  # 随机用灰色方框填充
        ])
        print(f'Using transform with random rotation, color jitter, and random resized crop')
    return chosen_transform

# 训练函数
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, start_epoch, device, best_acc, best_loss):
    # 使用日期作为目录名
    writer = SummaryWriter(log_dir=f'runs/log_{time.strftime("%Y%m%d_%H%M%S")}')
    
    best_val_loss = best_loss
    best_val_accuracy = best_acc
    print(f'pretrained model best_acc: {best_val_accuracy}, best_loss: {best_val_loss}')
    
    train_loader.dataset.transform = get_train_transform(False)
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # 显示图像和标签
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            img = images[i*4+j].permute(1, 2, 0).numpy()
            label = labels[i*4+j].numpy()
            label_str = vocab.sequence_to_text(label)
            axes[i][j].imshow(img)
            axes[i][j].set_title(label_str, fontproperties=prop)
            axes[i][j].axis('off')

    # 保存图像
    plt.savefig('batch_images.png')
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loader.dataset.transform = get_train_transform(epoch % 2 != 0)
        train_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        images_count = 0
        for images, tgt in train_loader_tqdm:
            images = images.to(device)
            tgt_input = tgt[:, :-1].to(device)
            tgt_output = tgt[:, 1:].to(device)

            optimizer.zero_grad()
            outputs = model(images, tgt_input)
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, tgt_output)
            loss.backward()
            optimizer.step()
            images_count += images.size(0)
            
            train_loss += loss.item() * images.size(0)
            train_loader_tqdm.set_postfix(loss=f"{train_loss / images_count:.8f}")
            train_loader_tqdm.update(1)

        train_loss = train_loss / images_count
        
        val_loss, val_accuracy, exact_match_accuracy = validate(model, val_loader, criterion, device)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('Accuracy/exact_match_val', exact_match_accuracy, epoch)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Val Accuracy: {val_accuracy:.4f}, Exact Match Accuracy: {exact_match_accuracy:.4f}')

        save_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best = True
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_best = True

        save_checkpoint(model, optimizer, epoch + 1, train_loss, val_accuracy, val_loss, filename=f'last_model.pth')
        
        if save_best:
            shutil.copyfile(os.path.join(checkpoints_folder, f'last_model.pth'), os.path.join(checkpoints_folder, f'best_model_{epoch+1}_{val_accuracy:.4f}_{exact_match_accuracy:.4f}_{val_loss:.8f}.pth'))
            print(f'Best model saved (epoch {epoch+1}, val_accuracy {val_accuracy:.4f}, val_loss {val_loss:.8f})')
    
    writer.close()
    
# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    exact_match_correct = 0
    total = 0
    images_count = 0
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

            # 使用掩码来计算正确的预测
            correct += (predicted[non_pad_mask] == labels[:, 1:][non_pad_mask]).sum().item()
            total += non_pad_mask.sum().item()

            # 计算每个序列的完全匹配情况
            exact_match_correct += ((predicted == labels[:, 1:]) | ~non_pad_mask).all(dim=1).sum().item()
            
            val_loader_tqdm.set_postfix(loss=f"{val_loss/images_count:.5f}")

    val_loss = val_loss / images_count
    accuracy = correct / total
    exact_match_accuracy = exact_match_correct / images_count
    return val_loss, accuracy, exact_match_accuracy

def save_checkpoint(model, optimizer, epoch, loss, best_acc, best_loss, filename):
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)
    
    print(f'Saving checkpoint {filename} (epoch {epoch}, loss {loss}, best_acc {best_acc}, best_loss {best_loss})')
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_acc': best_acc,
        'best_loss': best_loss
    }
    torch.save(state, os.path.join(checkpoints_folder, filename))

def load_checkpoint(model, optimizer, filename):
    state = torch.load(os.path.join(checkpoints_folder, filename))
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    epoch = state['epoch']
    loss = state['loss']
    if 'best_acc' in state:
        best_acc = state['best_acc']
    else:
        best_acc = 0.0
    if 'best_loss' in state:
        best_loss = state['best_loss']
    else:
        best_loss = float('inf')
    print(f'Loaded checkpoint {filename} (epoch {epoch}, loss {loss})')
    return model, optimizer, epoch, loss, best_acc, best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a license plate recognition model")
    parser.add_argument('--train_folder', type=str, default=DEFAULT_TRAIN_FOLDER, help='Path to the training dataset folder')
    parser.add_argument('--val_folder', type=str, default=DEFAULT_VAL_FOLDER, help='Path to the validation dataset folder')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Number of epochs to train')
    parser.add_argument('--checkpoint_path', type=str, default=DEFAULT_CHECKPOINT_PATH, help='Path to the model checkpoint file')
    args = parser.parse_args()
    
    # 创建数据集和数据加载器
    train_dataset = LicensePlateDataset(args.train_folder, vocab, max_length, get_train_transform(True))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_dataset = LicensePlateDataset(args.val_folder, vocab, max_length, get_train_transform(True))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size // 4, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LicensePlateModel(pad_idx=vocab.pad_idx, d_model=64, nhead_encoder=4, nhead_decoder=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, vocab_size=len(vocab.vocab_list), max_length=max_length)
    model.to(device)
    
    # 加载mobilenetv3预训练模型
    # model.backbone.load_state_dict(torch.load(r"C:\Users\mengchao\Downloads\450_act3_mobilenetv3_small.pth"), strict=False)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 加载检查点（如果存在）
    if args.checkpoint_path and os.path.exists(os.path.join(checkpoints_folder, args.checkpoint_path)):
        model, optimizer, start_epoch, train_loss, best_acc, best_loss = load_checkpoint(model, optimizer, args.checkpoint_path)
    else:
        start_epoch = 0
        best_loss = float('inf')
        best_acc = 0.0
        
    # val_loss, val_accuracy, exact_match_accuracy = validate(model, val_loader, criterion, device)
    # print(f'Val Loss: {val_loss:.8f}, Val Accuracy: {val_accuracy:.4f}, Exact Match Accuracy: {exact_match_accuracy:.4f}')
    
    train(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, start_epoch, device, best_acc, best_loss)
