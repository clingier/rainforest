import pytorch_lightning as pl
import torch
import pandas as pd

from torch.utils.data import Dataset
import soundfile as sf

from torch.utils.data import DataLoader

import numpy as np

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torch.nn.modules.pooling import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torch import nn
from torch.nn import BCEWithLogitsLoss

import timm
from functools import partial

train = pd.read_csv("./data/raw/RCSAD/train_tp.csv")
test = pd.read_csv("./data/raw/RCSAD/sample_submission.csv")
train.groupby("recording_id").agg(lambda x: list(x)).reset_index()

class TrainDataset(Dataset):
    def __init__(self, df, period=10, transforms=None, data_path="../data/raw/RCSAD/train", train=True):
        self.period = period
        self.transforms = transforms
        self.data_path = data_path
        self.train = train

        if train:
            dfgby = df.groupby("recording_id").agg(lambda x: list(x)).reset_index()
            self.recording_ids = dfgby["recording_id"].values
            self.species_ids = dfgby["species_id"].values
            self.t_mins = dfgby["t_min"].values
            self.t_maxs = dfgby["t_max"].values
        else:
            self.recording_ids = df["recording_id"].values
    
    def __len__(self):
        return len(self.recording_ids)

    def __getitem__(self, idx):
        recording_id = self.recording_ids[idx]
        if self.train:
            species_id = self.species_ids[idx]
            t_min, t_max = self.t_mins[idx], self.t_maxs[idx]
        else:
            species_id = [0]
            t_min, t_max = [0], [0]
        
        y, sr = sf.read(f"{self.data_path}/{recording_id}.flac")

        len_y = len(y)
        effective_length = sr * self.period
        rint = np.random.randint(len(t_min))
        tmin, tmax = round(sr * t_min[rint]), round(sr * t_max[rint])

        if len_y < effective_length:
            start = np.random.randint(effective_length - len_y)
            new_y = np.zeros(effective_length, dtype=y.dtype)
            new_y[start:start+len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            center = round((tmin + tmax) / 2)
            big = center - effective_length
            if big < 0:
                big = 0
            start = np.random.randint(big, center)
            y = y[start:start+effective_length]
            if len(y) < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start1 = np.random.randint(effective_length - len(y))
                new_y[start1:start1+len(y)] = y
                y = new_y.astype(np.float32)
            else:
                y = y.astype(np.float32)
        else:
            y = y.astype(np.float32)
            start = 0
        
        if self.transforms:
            y = self.transforms(samples=y, sample_rate=sr)
            
        start_time = start / sr
        end_time = (start + effective_length) / sr

        label = np.zeros(24, dtype='f')

        if self.train:
            for i in range(len(t_min)):
                if (t_min[i] >= start_time) & (t_max[i] <= end_time):
                    label[species_id[i]] = 1
                elif start_time <= ((t_min[i] + t_max[i]) / 2) <= end_time:
                    label[species_id[i]] = 1
        
        return {
            "waveform" : y,
            "target" : torch.tensor(label, dtype=torch.float),
            "id" : recording_id
        }

class TestDataset(Dataset):
    def __init__(self, df, period=10, transforms=None, data_path="../data/raw/RCSAD/train", train=True):
        self.period = period
        self.transforms = transforms
        self.data_path = data_path
        self.train = train
        
        self.recording_ids = df["recording_id"].values

    
    def __len__(self):
        return len(self.recording_ids)
    
    def __getitem__(self, idx):

        recording_id = self.recording_ids[idx]
        
        y, sr = sf.read(f"{self.data_path}/{recording_id}.flac")
        
        len_y = len(y)
        effective_length = sr * self.period
        
        y_ = []
        i = 0
        while i < len_y:
            y__ = y[i:i+effective_length]
            
            if self.transforms:
                y__ = self.transforms(samples=y__, sample_rate=sr)
                
            y_.append(y__)
            i = i + effective_length
        
        y = np.stack(y_)

        label = np.zeros(24, dtype='f')
        
        return {
            "waveform" : y,
            "target" : torch.tensor(label, dtype=torch.float),
            "id" : recording_id
        }

class AudioDataset(pl.LightningDataModule):
    def __init__(self, train_df, test_df, data_path="./data/raw/RCSAD/train" ,train_batch_size=16, num_workers=2):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        dataset = TrainDataset(train_df, data_path=data_path)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
        self.test_dataset = TestDataset(test_df, data_path=data_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.train_batch_size)

encoder_params = {
    "resnet50d" : {
        "features" : 2048,
        "init_op"  : partial(timm.models.resnest50d, pretrained=True, in_chans=1)
    },
}

import torchmetrics

def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out

class AudioClassifier(nn.Module):
    def __init__(self,
            encoder="resnet50d",
            sample_rate=48_000,
            window_size=2048,
            hop_size=512,
            mel_bins=128,
            fmin=0,
            fmax=24_000,
            classes_num=24
        ):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        #self.max_pool = AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(encoder_params[encoder]['features'], classes_num)
    
    def forward(self, input, spec_aug=False, mixup_lambda=None):
        #print(input.type())
        x = self.spectrogram_extractor(input.float()) # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x) # (batch_size, 1, time_steps, mel_bins)

        #if spec_aug:
        #    x = self.spec_augmenter(x)
        if self.training:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            #pass
        
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class AudioClassifierModule(pl.LightningModule):

    def __init__(
            self,
            loss_fn=None,
            learning_rate=1e-5,
            model_parameters=None,
            train_dl=None,
        ):
        super(AudioClassifierModule, self).__init__()
        self.model = AudioClassifier(**model_parameters) if model_parameters else AudioClassifier()
        self.learning_rate = learning_rate
        self.loss = loss_fn
        self.metrics = [torchmetrics.Accuracy()]
        self.train_dl = train_dl

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def forward(self,x):
            logits = self.model(x)
            return logits

    def _step(self, batch, stage=None):
        if stage != "test":
            output = self.model(batch['waveform'])        
            loss = self.loss(output, batch['target'])

            # for m in self.metrics:
            #     m.update(batch['target'], output)
            return {'loss': loss}
        else:
            pred_list = []
            id_list = []
            for x in batch[0]:
                input = x["waveform"]
                bs, seq, w = input.shape
                input = input.reshape(bs*seq, w)
                id = x["id"]
                output = torch.sigmoid(self.model(input))
                output = output.reshape(bs, seq, -1)
                output, _ = torch.max(output, dim=1)
                output = output.cpu().detach().numpy().tolist()
                pred_list.extend(output)
                id_list.extend(id)
            return {"preds": pred_list, "id": id_list}
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")
    
dataset = AudioDataset(train, test)

trainer = pl.Trainer(accelerator="gpu", devices=1)
train_dl = dataset.train_dataloader()
val_dl = dataset.val_dataloader()
litmodel = AudioClassifierModule(loss_fn=BCEWithLogitsLoss(), train_dl=train_dl)

trainer.fit(litmodel, train_dl, val_dl)