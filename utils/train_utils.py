### Heavily inspired by facebook projects with same format

import monai
import torch
from sklearn.metrics import accuracy_score, precision_score


def train_one_epoch(net, trn_dataloader, criterion, optimizer, scheduler=None, grad_clip=None):
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trn_dataloader):
        optimizer.zero_grad()

        inputs, labels = data["input"], data["label"]

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backwards()

        running_loss += loss.item()

        if grad_clip:
            torch.nn.utils.clip_grad_norm(net.parameters(), grad_clip)

        optimizer.step()

    if scheduler:
        scheduler.step()

    return running_loss / len(trn_dataloader)


def evaluate(net, val_dataloader, metrics=None):
    if metrics is None:
        metrics = ["accuracy"]

    incl_metrics = ["accuracy", "precision", "recall", "dice", "iou"]
    for metric in metrics:
        assert metric in incl_metrics

    metric_results = {}

    net.eval()
    for i, data in enumerate(val_dataloader):

        inputs, labels = data["input"], data["label"]

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

        outputs = net(inputs)

        for metric in metrics:
            if metric == "accuracy":
                metric_results[metric] += accuracy_score(torch.argmax(outputs, dim=1).flatten(3),
                                                         labels.flatten(3)) / len(val_dataloader)
            if metric == "precision":
                metric_results[metric] += precision_score(torch.argmax(outputs, dim=1).flatten(3),
                                                          labels.flatten(3)) / len(val_dataloader)
            if metric == "recall":
                metric_results[metric] += precision_score(torch.argmax(outputs, dim=1).flatten(3),
                                                          labels.flatten(3)) / len(val_dataloader)
            if metric == "dice":
                metric_results[metric] += monai.metrics.compute_meandice(outputs, labels) / len(val_dataloader)
            if metric == "iou":
                metric_results[metric] += monai.metrics.compute_meaniou(outputs, labels) / len(val_dataloader)

        return metric_results
