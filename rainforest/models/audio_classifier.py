from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torch.nn.modules.pooling import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torch import nn

from sklearn.metrics import confusion_matrix, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef

import torch

import timm
from functools import partial

import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt

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
            model_parameters=None
        ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AudioClassifier(**model_parameters) if model_parameters else AudioClassifier()
        self.learning_rate = learning_rate
        self.loss = loss_fn
        self.metrics = [torchmetrics.Accuracy()]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


    def forward(self,x):
            logits = self.model(x)
            return logits

    def _step(self, batch, stage=None):
        if stage != "test":
            output = self.model(batch['waveform'])        
            loss = self.loss(output, batch['target'])
            return {'loss': loss}

        else:
            pred_list = []
            input = batch['waveform']
            seq, w = input.shape
            input = input.reshape(seq, w)
            output = torch.sigmoid(self.model(input))
            output = output.cpu().detach()
            return {"preds": output, "target": batch["target"].cpu().detach()}
            
    def _epoch_end(self, outputs, stage=None):
        if stage != "test":
            loss = np.mean([float(x['loss']) for x in outputs])
            self.logger.log_metrics(
                {f"{stage}/loss": loss}, self.current_epoch + 1
            )
        if stage == "test":
            preds = outputs[0]["preds"]
            targets = outputs[0]["target"]
            for i in range(1, len(outputs)):
                preds = torch.vstack((preds, outputs[i]["preds"]))
                targets = torch.vstack((targets, outputs[i]["target"]))
            
            preds_argmax = torch.argmax(preds, dim=1)
            targets_argmax = torch.argmax(targets, dim=1)

            precision = precision_score(targets_argmax.numpy(), preds_argmax.numpy(), average='macro')
            recall = recall_score(targets_argmax.numpy(), preds_argmax.numpy(), average='macro')
            matthews_corr = matthews_corrcoef(targets_argmax.numpy(), preds_argmax.numpy())
            cohen_kappa = cohen_kappa_score(targets_argmax.numpy(), preds_argmax.numpy())

            matrix = confusion_matrix( targets_argmax.numpy(), preds_argmax.numpy(),)
            print(matrix)
            
           

            fig, ax = plt.subplots(figsize=(20, 20))

            ax.matshow(matrix, cmap=plt.cm.Blues)

            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[0]):
                    c = matrix[j,i]
                    ax.text(i, j, str(c), va='center', ha='center')
            plt.savefig("outputs/figures/confusion_matrix.png")

            acc = torch.sum(preds_argmax == targets_argmax) / preds_argmax.shape[0]
            self.log("test_acc", acc)
            self.log("test_precision", precision)
            self.log("test_recall", recall)
            self.log("test_matthews_corr", matthews_corr)
            self.log("test_cohen_kappa", cohen_kappa)

    
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")
    
    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "test")