from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl

import utils

class SingleView(pl.LightningModule):
    def __init__(self, hparams):
        super(SingleView, self).__init__()
        self.hparams = hparams

    def forward(self, x):
        return x

    def training_step(self, batch, _):
        images, target = batch
        output = self.forward(images)
        loss_val = F.cross_entropy(output, target)

        acc = utils.accuracy(output, target)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'acc': acc,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def validation_step(self, batch, _):
        images, target = batch
        output = self.forward(images)
        loss_val = F.cross_entropy(output, target)
        acc = utils.accuracy(output, target)

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': acc,
        })

        return output

    def validation_epoch_end(self, outputs):
        tqdm_dict = {}

        for metric_name in ["val_loss", "val_acc"]:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]
                metric_total += metric_value

            tqdm_dict[metric_name] = metric_total / len(outputs)

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
