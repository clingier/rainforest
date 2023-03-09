from torch.nn import BCEWithLogitsLoss

from rich import print, pretty, inspect, traceback
pretty.install()
traceback.install()

from rainforest.models.audio_classifier import AudioClassifierModule
from rainforest.data.datasets import AudioDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd

class args:
    train_df_path = "./data/raw/RCSAD/train_tp.csv"
    test_df_path = "./data/raw/RCSAD/sample_submission.csv"
    checkpoint_dir = "./outputs/checkpoints"
    loss_fn = BCEWithLogitsLoss()
    train_batch_size = 16
    num_workers = 3
    learning_rate=1e-5
    model_parameters=None

train = pd.read_csv(args.train_df_path)
test = pd.read_csv(args.test_df_path)
train.groupby("recording_id").agg(lambda x: list(x)).reset_index()

pl.seed_everything(42)

ds = AudioDataset(
    train,
    test,
    train_batch_size=args.train_batch_size,
    num_workers=args.num_workers
)

log_dir = "./outputs/logs/tb_logs"
tb_logger = TensorBoardLogger(save_dir=log_dir)

mod = AudioClassifierModule(
    loss_fn=args.loss_fn,
    learning_rate=args.learning_rate,
    model_parameters=args.model_parameters
)

checkpoint_callback = ModelCheckpoint(
    dirpath=args.checkpoint_dir,
    monitor=None,
    verbose=True
)

trainer = pl.Trainer(
    logger=tb_logger,
    accelerator="gpu",
    devices=1,
    callbacks=[checkpoint_callback],
    max_epochs=150
)

trainer.fit(mod, ds.train_dataloader(), ds.val_dataloader())