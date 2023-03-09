import pytorch_lightning as pl

from rainforest.models import AudioClassifierModule
from rainforest.data.datasets import AudioDataset

from rich import print, pretty, inspect, traceback
pretty.install()
traceback.install()



import pandas as pd

class args:
    train_df_path = "./data/raw/RCSAD/train_tp.csv"
    test_df_path = "./data/raw/RCSAD/sample_submission.csv"
    checkpoint_dir = "./outputs/checkpoints"
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

model = AudioClassifierModule.load_from_checkpoint("outputs\checkpoints\epoch=149-step=8549.ckpt")

val_dl = ds.val_dataloader()

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=150
)

trainer.test(model, val_dl)