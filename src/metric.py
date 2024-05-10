from torchmetrics import Metric
import torch
import numpy as np

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("TP", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("FP", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("FN", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape, "Predictions and targets must have the same shape"

        for cls in range(self.num_classes):
            true_cls = target == cls
            pred_cls = preds == cls
 
            self.TP[cls] += torch.logical_and(pred_cls, true_cls).sum()
            self.FP[cls] += torch.logical_and(pred_cls, torch.logical_not(true_cls)).sum()
            self.FN[cls] += torch.logical_and(torch.logical_not(pred_cls), true_cls).sum()

    def compute(self):
        precision = self.TP / (self.TP + self.FP + 1e-6)
        recall = self.TP / (self.TP + self.FN + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        return f1_score.mean()
        
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, axis=1)

        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, "Predictions and targets must have the same shape"
 
        # [TODO] Count the number of correct prediction
        correct = torch.sum(preds == target).item()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):       
        return self.correct.float() / self.total.float()
