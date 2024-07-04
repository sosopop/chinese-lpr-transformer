import torch
import torch.nn as nn
import torchvision.transforms as transforms
from mobilenetv3 import MobileNetV3_Small

# 定义2D位置编码类
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(d_model, height, width)
        y_position = torch.arange(0, height, dtype=torch.float).unsqueeze(1).unsqueeze(2)
        x_position = torch.arange(0, width, dtype=torch.float).unsqueeze(0).unsqueeze(2)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[0::2, :, :] = torch.sin(y_position * div_term).permute(2, 0, 1)
        pe[1::2, :, :] = torch.cos(y_position * div_term).permute(2, 0, 1)

        pe[0::2, :, :] += torch.sin(x_position * div_term).permute(2, 0, 1)
        pe[1::2, :, :] += torch.cos(x_position * div_term).permute(2, 0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(2), :x.size(3)]

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_length):
        super(PositionalEncoding1D, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ImageEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(ImageEncoder, self).__init__()
        self.backbone = MobileNetV3_Small()
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-6])  # 去掉池化前的最后6层
        
        self.conv1 = nn.Conv2d(576, d_model, kernel_size=1, stride=1, padding=0, bias=False)
        self.positional_2d_encoding = PositionalEncoding2D(d_model, height=7, width=7)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
                ), 
            num_layers=num_layers)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.positional_2d_encoding(x)
        x = x.flatten(2).permute(0, 2, 1)
        memory = self.transformer_encoder(x)
        return memory

class TextDecoder(nn.Module):
    def __init__(self, pad_idx, vocab_size, d_model=128, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_length=16):
        super(TextDecoder, self).__init__()
        self.pad_idx = pad_idx
        self.positional_1d_encoding = PositionalEncoding1D(d_model, max_length)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
                ),
            num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, memory, tgt):
        seq_length_tgt = tgt.size(1)
        tgt_emb = self.positional_1d_encoding(self.embedding(tgt))
        tgt_mask = self.generate_square_subsequent_mask(seq_length_tgt).to(tgt.device)
        tgt_padding_mask = (tgt == self.pad_idx).to(tgt.device)
        
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        output = self.fc(output)
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class LicensePlateModel(nn.Module):
    def __init__(self, pad_idx, vocab_size, d_model=128, nhead_encoder=8, nhead_decoder=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_length=16):
        super(LicensePlateModel, self).__init__()
        
        self.image_encoder = ImageEncoder(d_model=d_model, nhead=nhead_encoder, num_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.text_decoder = TextDecoder(pad_idx=pad_idx, vocab_size=vocab_size, d_model=d_model, nhead=nhead_decoder, num_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, max_length=max_length)
        
    def forward(self, x, tgt):
        memory = self.image_encoder(x)
        output = self.text_decoder(memory, tgt)
        return output
    
    def encode_image(self, x):
        memory = self.image_encoder(x)
        return memory
    
    def decode_text(self, memory, tgt):
        output = self.text_decoder(memory, tgt)
        return output

if __name__ == '__main__':
    from license_plate_dataset import LicensePlateDataset, LicensePlateVocab
    from torch.utils.data import DataLoader
    
    img_height = 224
    img_width = 224
    
    # 设置数据变换
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
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
    
    
    model = LicensePlateModel(pad_idx=vocab.pad_idx, vocab_size=vocab.vocab_size, max_length=max_length)
    output = model(images, labels)
    print(output.shape)  # [4, 10, 10]