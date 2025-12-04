import torch
import torchaudio
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
annotation = r"C:\Users\Nikhil Pathak\OneDrive\Desktop\Deep Learning\Urban-sound-model\UrbanSound8K.csv"
audio_dir = r"C:\Users\Nikhil Pathak\OneDrive\Desktop\Deep Learning\Urban-sound-model\UrbanSound8K\audio"


class AudioDataset(Dataset):
    def __init__(self, annotation, audio_dir, transformation, target_sampling_rate, num_samples, device):
        self.annotation = pd.read_csv(annotation)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sampling_rate = target_sampling_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sampling_rate)
            resampler = resampler.to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        if signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotation.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotation.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotation.iloc[index, 6]



# Audio parameters
sample_rate = 22050
num_samples = 22050

# Mel Spectrogram
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

usd = AudioDataset(annotation, audio_dir, mel_spectrogram, sample_rate, num_samples, device)

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ResidualBlockWithAttention(nn.Module):
    """Residual Block with CBAM Attention"""
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Attention module
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply attention
        if self.use_attention:
            out = self.cbam(out)

        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (Alternative attention)"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionResNet(nn.Module):
    """ResNet with integrated CBAM Attention for Audio Classification"""
    def __init__(self, num_classes=10, attention_type='cbam'):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual layers with attention
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global context
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # First block (may downsample)
        layers.append(ResidualBlockWithAttention(in_channels, out_channels, stride, use_attention=True))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlockWithAttention(out_channels, out_channels, 1, use_attention=True))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Lighter version for faster training
class LightAttentionResNet(nn.Module):
    """Lighter version with fewer parameters"""
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(32, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlockWithAttention(in_channels, out_channels, stride, use_attention=True))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlockWithAttention(out_channels, out_channels, 1, use_attention=True))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = AttentionResNet(num_classes=10).to(device)

print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

from torchsummary import summary
summary(model, (1, 64, 44))

state_dict = torch.load("/content/drive/MyDrive/Urban_Sound.pth", map_location=device)
model.load_state_dict(state_dict['model_state_dict'])

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    'dog_bark',
    'drilling',
    'engine_idling',
    'gun_shot',
    'jackhammer',
    'siren',
    'street_music'
]

def predict(model,input,target,class_mapping):
  model.eval()
  with torch.no_grad():
    prediction = model(input)
    predicted_index = prediction[0].argmax(0)
    predicted_class = class_mapping[predicted_index]
    expected = class_mapping[target]

  return predicted_class,expected

input,target = usd[50][0] , usd[50][1]
input.unsqueeze_(0)
predicted,expected = predict(model,input,target,class_mapping)
print(f"The expected output is: {expected}, and the predicted is: {predicted}")
