import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torcheval.metrics.functional import multiclass_f1_score


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy_triplet(pred_target, target):
    with torch.no_grad():
        correct = 0
        for i in range(len(pred_target)):
            if pred_target[i].item() == target[i].item():
                correct += 1
    return correct / len(pred_target)


def f1_score(output, target, num_classes):
    #pred = torch.argmax(output, dim=1)
    return multiclass_f1_score(output, target, num_classes=num_classes)


def balanced_accuracy(pred_target, target):
    with torch.no_grad():
        pred = torch.argmax(pred_target, dim=1)
        assert pred.shape[0] == len(target)
        return balanced_accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def triplet_loss(anchor, positive, negative, loss_paras=None):
    if loss_paras is None:
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    else:
        triplet_loss = nn.TripletMarginLoss(**loss_paras)
    return triplet_loss(anchor, positive, negative)
