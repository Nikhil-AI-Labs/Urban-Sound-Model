import torch
import torchaudio
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import os
from google.colab import drive
from torchsummary import summary
drive.mount('/content/drive')

annotation = "/content/drive/MyDrive/UrbanSound8K/metadata/UrbanSound8K.csv"
audio_dir = "/content/drive/MyDrive/UrbanSound8K/audio" #/content/drive/MyDrive/UrbanSound8K/audio

class AudioDataset(Dataset):
  def __init__(self,annoation,audio_dir,transformation,target_sampling_rate,num_samples,device):
    self.annoation = pd.read_csv(annoation)
    self.audio_dir = audio_dir
    self.device = device
    self.transformation = transformation.to(self.device)
    self.target_sampling_rate = target_sampling_rate
    self.num_samples = num_samples

  def __len__(self):
    return len(self.annoation)

  def __getitem__(self, index):
    audio_sample_path = self._get_audio_sample_path(index)
    label = self._get_audio_sample_label(index)
    signal,sr = torchaudio.load(audio_sample_path)
    signal = signal.to(self.device)
    signal = self._resample_if_necessary(signal,sr)
    signal = self._mix_down_if_necessary(signal)
    signal = self._cut_if_necessary(signal)
    signal = self._right_pad_if_necessary(signal)
    signal = self.transformation(signal)
    return signal,label

  def _resample_if_necessary(self, signal, sr):
    if sr != self.target_sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, self.target_sampling_rate)
        # Move the resampler's internal parameters to the device
        resampler = resampler.to(self.device)
        signal = resampler(signal)
    return signal


  def _mix_down_if_necessary(self,signal):
    if signal.shape[0]>1:
      signal = torch.mean(signal,dim=0,keepdim=True)

    return signal


  def _cut_if_necessary(self,signal):
    if signal.shape[1] > self.num_samples:
      signal = signal[:,:self.num_samples]

    return signal

  def _right_pad_if_necessary(self,signal):
    if signal.shape[1] < self.num_samples:
      num_missing_samples = self.num_samples - signal.shape[1]
      last_dim_padding = (0,num_missing_samples)
      signal = torch.nn.functional.pad(signal,last_dim_padding)

    return signal

  def _get_audio_sample_path(self,index):
    fold = f"fold{self.annoation.iloc[index,5]}"
    path = os.path.join(self.audio_dir,fold,self.annoation.iloc[index,0])
    return path

  def _get_audio_sample_label(self,index):
    return self.annoation.iloc[index,6]

# from torch.cuda import is_available
sample_rate = 22050
num_samples = 22050
if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

print(f"Using {device} device")

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

usd = AudioDataset(annotation,audio_dir,mel_spectrogram,sample_rate,num_samples,device)


class CNNnetwork(nn.Module):
  def __init__(self, in_channel):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channel,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv2 = nn.Sequential(
        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv3 = nn.Sequential(
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv4 = nn.Sequential(
        nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128*5*4,256),
        nn.ReLU(),
        nn.Linear(256,10),
        nn.Softmax(dim=1)
    )

  def forward(self,x):
    return self.model(self.conv4(self.conv3(self.conv2(self.conv1(x)))))

cnn = CNNnetwork(1).to(device)
summary(cnn,(1,64,44))

batch_size = 128
epoch = 30
lr = 0.001

train_loader = DataLoader(usd, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(),lr=lr)
for i in range(epoch):
  print(f"Epoch: {i+1}")
  for input,target in train_loader:
    target = target.to(device)
    out = cnn(input)
    loss = loss_fn(out,target)
    optimizer.zero_grad() # Clear gradients before backward pass
    loss.backward()
    optimizer.step()

  print(f"loss: {loss.item()}")
  print("-"*20)

print("Finished Training")
torch.save(cnn.state_dict(),"Urban_Sound.pth")
print("File saved successfully")

torch.save(cnn.state_dict(), "/content/drive/MyDrive/Urban_Sound.pth")
print("Model saved successfully to Google Drive")

