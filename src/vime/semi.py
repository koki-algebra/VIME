from typing import List
import torch
from torch.optim import Optimizer
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from utils.pretext_generator import pretext_generator


class SemiSLNetworks(nn.Module):
    def __init__(self, encoder: nn.Module, dim_z: int, dim_y: int) -> None:
        super().__init__()
        # pre-trained encoder
        self.encoder = encoder
        # predictor
        self.predictor = nn.Sequential(
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_z*2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_z*2),
            nn.Linear(dim_z*2, dim_z),
            nn.ReLU(),
            nn.Linear(dim_z, dim_y)
        )

    def forward(self, X: Tensor) -> Tensor:
        Z = self.encoder(X)
        logits = self.predictor(Z)
        return logits


class SemiSLLoss(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, y_pred: Tensor, y_target: Tensor, unlabeled_y_tilde_pred: Tensor, unlabeled_y_pred: Tensor) -> float:
        supervised_loss = nn.CrossEntropyLoss()
        unsupervised_loss = nn.MSELoss()

        return supervised_loss(y_pred, y_target) + self.beta * unsupervised_loss(unlabeled_y_tilde_pred, unlabeled_y_pred)


def semi_train(labeled_dataloader: DataLoader, unlabeled_dataloader: DataLoader, model: SemiSLNetworks, loss_fn: SemiSLLoss, optimizer: Optimizer, p_m: float, K: int) -> None:
    size = len(labeled_dataloader.dataset)
    model.train()
    for batch, (X_l, y) in enumerate(labeled_dataloader):
        # labeled
        y_l_pred = model(X_l)

        # unlabeled
        X_u = next(iter(unlabeled_dataloader))

        # K times Data Augmentation
        y_u_tilde_pred_list: List[Tensor] = []
        y_u_pred_list: List[Tensor] = []
        for _ in range(K):
            _, X_u_tilde = pretext_generator(X=X_u, p_m=p_m)

            # predict
            y_u_tilde_pred = model(X_u_tilde)
            y_u_pred = model(X_u)
            # append
            y_u_tilde_pred_list.append(y_u_tilde_pred)
            y_u_pred_list.append(y_u_pred)

        y_u_tilde_pred = torch.cat(y_u_tilde_pred_list)
        y_u_pred = torch.cat(y_u_pred_list)

        # loss
        loss = loss_fn(y_l_pred, y, y_u_tilde_pred, y_u_pred)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            current = batch * len(X_l)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def semi_test(dataloader: DataLoader, model: SemiSLNetworks, loss_fn: nn.CrossEntropyLoss):
    size = len(dataloader.dataset) # type: ignore
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y_logits = model(X)
            y_prob = softmax(input=y_logits, dim=1)

            test_loss += loss_fn(y_logits, y).item()
            correct += (y_prob.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
