# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

# Custom packages
from src.metric import MyAccuracy, MyF1Score
import src.config as cfg
from src.util import show_setting
from src.network_rev import MyNetwork_rev

# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):
    def __init__(self, num_classes=200, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.LeakyReLU(0.01),  
            nn.BatchNorm1d(4096),  
            nn.Dropout(dropout),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
                 dropout_rate: float = 0.5
        ):
        super().__init__()

        # Network
        if model_name == 'MyNetwork':
            self.model = MyNetwork(num_classes=num_classes, dropout=dropout_rate)

        
        elif model_name == 'MyNetwork_rev':
            self.model = MyNetwork_rev(num_classes=num_classes, dropout=dropout_rate)
            
        # elif model_name == 'SOTA2':
        #     self.model = SOTA(num_classes=num_classes, dropout=dropout_rate)
        
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)
            # num_ftrs = self.model.fc.in_features  
            # #Regulation
            # self.model.fc = nn.Sequential(
            #     nn.Dropout(dropout_rate),  
            #     nn.Linear(num_ftrs, num_classes)
            # )
        

        # Loss function
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        # self.loss_fn = nn.NLLLoss()  # Negative Log-Likelihood Loss

        # Metric
        self.accuracy = MyAccuracy() 
        self.f1_score = MyF1Score(num_classes)
        
        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)
        # return F.log_softmax(x, dim=1)

    # def training_step(self, batch, batch_idx):
    #     loss, scores, y = self._common_step(batch)
    #     accuracy = self.accuracy(scores, y)
    #     self.log_dict({'loss/train': loss, 'accuracy/train': accuracy},
    #                   on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     loss, scores, y = self._common_step(batch)
    #     accuracy = self.accuracy(scores, y)
    #     self.log_dict({'loss/val': loss, 'accuracy/val': accuracy},
    #                   on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    # def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
    #     if not isinstance(self.logger, WandbLogger):
    #         if batch_idx == 0:
    #             self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
    #         return

    #     if batch_idx % frequency == 0:
    #         x, y = batch
    #         preds = torch.argmax(preds, dim=1)
    #         self.logger.log_image(
    #             key=f'pred/val/batch{batch_idx:5d}_sample_0',
    #             images=[x[0].to('cpu')],
    #             caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])


    def training_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1_score(scores, y)
        self.log_dict({
            'loss/train': loss,
            'accuracy/train': accuracy,
            'f1_score/train': f1
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        accuracy = self.accuracy(scores, y)
        f1 = self.f1_score(scores, y)
        self.log_dict({
            'loss/val': loss,
            'accuracy/val': accuracy,
            'f1_score/val': f1
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores)

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return
        if batch_idx % cfg.WANDB_IMG_LOG_FREQ == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:05d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])