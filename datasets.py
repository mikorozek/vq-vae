import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchaudio.datasets import VCTK_092

class VCTKFixed(Dataset):
    def __init__(self, root, download=False, sample_rate=48000, seconds=4):
        self.dataset = VCTK_092(root=root, download=download)
        self.target_length = seconds * sample_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, _, _, _, _ = self.dataset[index]
        
        current_length = waveform.shape[1]

        if current_length < self.target_length:
            padding = self.target_length - current_length
            waveform = F.pad(waveform, (0, padding))
        elif current_length > self.target_length:
            waveform = waveform[:, :self.target_length]

        if torch.max(torch.abs(waveform)) > 0:
            waveform = waveform / torch.max(torch.abs(waveform))

        return waveform
