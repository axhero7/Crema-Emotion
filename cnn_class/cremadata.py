from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
import os
import numpy as np

class CremaSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device) -> None:
        super().__init__()
        if isinstance(annotations_file, str):  # Accept both path or DataFrame
            self.annotations = pd.read_csv(annotations_file)
        else:
            self.annotations = annotations_file
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label, int(self.annotations.iloc[index]["Filename"][:4])
    
    def get_with_id(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self.transformation(signal)
        # print(audio_sample_path.split("\\")[1][:4])
        return signal, label
    
    def _get_audio_sample_path(self, index):
        filename = self.annotations.iloc[index]["Filename"]
        path = os.path.join(self.audio_dir, filename + ".wav")
        return path
    
    def _get_audio_sample_label(self, index):
        nametoclass = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG":3, "FEA":4, "DIS":5}
        return nametoclass[self.annotations.loc[index, "Filename"].split("_")[-2]]
    
    def _resample_if_necessary(self, signal, sr):
        if sr == self.target_sample_rate:
            return signal
        resampled = torchaudio.transforms.Resample(sr, self.target_sample_rate)
        signal = resampled(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] < 1:
            return signal
        signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = 0, num_missing_samples
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def create_data_loader(dataset, batch_size, train_ratio, device):
        train_ratio = 0.8
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

        torch.manual_seed(42)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
        return train_dataloader, test_dataloader
   

