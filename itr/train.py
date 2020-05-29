import time
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from model import build_model
from data import PadSequence, IndicDataset
from config import replace, preEnc, preEncDec

def preproc_data():

    split_data('/content/drive/My Drive/Offnote labs/data/hin-eng/hin.txt', '/content/drive/My Drive/Offnote labs/data/hin-eng')

def gen_model_loaders(config):
    model, tokenizers = build_model(config)

    pad_sequence = PadSequence(tokenizers.src.pad_token_id, tokenizers.tgt.pad_token_id)

    train_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, True), 
                            batch_size=config.batch_size, 
                            shuffle=False, 
                            collate_fn=pad_sequence)
    eval_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, False), 
                           batch_size=config.eval_size, 
                           shuffle=False, 
                           collate_fn=pad_sequence)
    return model, tokenizers, train_loader, eval_loader

rconf = preEncDec
preproc_data()
model, tokenizers, train_loader, eval_loader = gen_model_loaders(rconf)
logger = TensorBoardLogger("/content/drive/My Drive/Offnote labs/tb_logs")
trainer = pl.Trainer( max_nb_epochs=10,gpus=[0], logger=logger)
trainer.fit(model, train_dataloader=train_loader)