import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

class LicensePlateVocab:
    def __init__(self, vocab_list, pad_token='#', eos_token='$', bos_token='^'):
        self.vocab_list = vocab_list + [pad_token, eos_token, bos_token]
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.vocab_dict = {char: idx for idx, char in enumerate(self.vocab_list)}
        self.idx_dict = {idx: char for idx, char in enumerate(self.vocab_list)}
        self.pad_idx = self.vocab_dict[pad_token]
        self.eos_idx = self.vocab_dict[eos_token]
        self.bos_idx = self.vocab_dict[bos_token]
        self.vocab_size = len(self.vocab_list)

    def text_to_sequence(self, text, max_length, pad_to_max_length=True, add_eos=True, add_bos=True):
        sequence = []
        if add_bos:
            sequence.append(self.bos_idx)  # Add BOS token at the beginning
        for char in text:
            if char in self.vocab_dict:
                sequence.append(self.vocab_dict[char])
        if add_eos:
            sequence.append(self.eos_idx)  # Add EOS token at the end
        if len(sequence) < max_length:
            if pad_to_max_length:
                sequence = sequence + [self.pad_idx] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return sequence

    def sequence_to_text(self, sequence):
        return ''.join([self.idx_dict[idx] for idx in sequence if idx != self.pad_idx and idx != self.eos_idx and idx != self.bos_idx])

class LicensePlateDataset(Dataset):
    def __init__(self, image_folder, vocab, max_length=16, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 提取车牌号
        plate_number = img_name.split('-')[0]
        label = self.vocab.text_to_sequence(plate_number, self.max_length)

        return image, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # 设置数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 词汇表
    vocab_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']
    vocab = LicensePlateVocab(vocab_list)

    # 最大序列长度
    max_length = 16  # 适当增加以包含EOS和可能的PAD

    # 创建数据集和数据加载器
    train_folder = r'D:\code\transformer_plate\datasets\train'
    val_folder = r'D:\code\transformer_plate\datasets\val'

    train_dataset = LicensePlateDataset(train_folder, vocab, max_length, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # 设置中文字体
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 可以选择其他中文字体
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_path]
    plt.rcParams['axes.unicode_minus'] = False

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
    plt.show()
