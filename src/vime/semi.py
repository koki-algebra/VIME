from typing import List, Dict
import torch
from torch.optim import Optimizer, Adam
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax

from .pretext import pretext_generator


class SemiSLNetworks(nn.Module):
    def __init__(self, encoder: nn.Module, dim_z: int, dim_y: int) -> None:
        super().__init__()
        # pre-trained encoder
        self.encoder = encoder
        # predictor
        self.predictor = nn.Sequential(
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_z * 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_z * 2),
            nn.Linear(dim_z * 2, dim_z),
            nn.ReLU(),
            nn.Linear(dim_z, dim_y),
        )

    def forward(self, X: Tensor) -> Tensor:
        Z = self.encoder(X)
        logits = self.predictor(Z)
        return logits

    def fit(
        self,
        labeled_dataset: Dataset,
        unlabeled_dataset: Dataset,
        test_dataset: Dataset,
        hyperparams: Dict[str, float | int],
        device: str,
    ) -> None:
        labeled_batch_size: int = hyperparams["labeled_batch_size"]
        unlabeled_batch_size: int = hyperparams["unlabeled_batch_size"]
        learning_rate: float = hyperparams["lr"]
        epochs: float = hyperparams["epochs"]
        p_m: float = hyperparams["p_m"]
        beta: float = hyperparams["beta"]
        K: int = hyperparams["K"]

        if beta < 0.0:
            raise ValueError("beta must be greater than 0.0")
        if K < 0:
            raise ValueError("K must be grater than 0")

        # model
        model = self.to(device)
        # loss function
        loss_fn = SemiSLLoss(beta)
        test_loss_fn = torch.nn.CrossEntropyLoss()
        # optimizer
        optimizer = Adam(params=model.parameters(), lr=learning_rate)

        # data loader
        labeled_dataloader = DataLoader(
            dataset=labeled_dataset, batch_size=labeled_batch_size
        )
        unlabeled_dataloader = DataLoader(
            dataset=unlabeled_dataset, batch_size=unlabeled_batch_size
        )
        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=labeled_batch_size
        )

        # fix encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # training & test
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(
                labeled_dataloader=labeled_dataloader,
                unlabeled_dataloader=unlabeled_dataloader,
                loss_fn=loss_fn,
                model=model,
                optimizer=optimizer,
                device=device,
                p_m=p_m,
                K=K,
            )
            test(
                dataloader=test_dataloader,
                model=model,
                loss_fn=test_loss_fn,
                device=device,
            )
        print("Done!")


class SemiSLLoss(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        y_pred: Tensor,
        y_target: Tensor,
        unlabeled_y_tilde_pred: Tensor,
        unlabeled_y_pred: Tensor,
    ) -> float:
        supervised_loss = nn.CrossEntropyLoss()
        unsupervised_loss = nn.MSELoss()

        return supervised_loss(y_pred, y_target) + self.beta * unsupervised_loss(
            unlabeled_y_tilde_pred, unlabeled_y_pred
        )


def train(
    labeled_dataloader: DataLoader,
    unlabeled_dataloader: DataLoader,
    model: SemiSLNetworks,
    loss_fn: SemiSLLoss,
    optimizer: Optimizer,
    device: str,
    p_m: float,
    K: int,
) -> None:
    size = len(labeled_dataloader.dataset)
    model.train()
    for batch, ((X_l, y), X_u) in enumerate(
        zip(labeled_dataloader, unlabeled_dataloader)
    ):
        X_l: Tensor = X_l.to(device)
        y: Tensor = y.to(device)
        X_u: Tensor = X_u.to(device)

        # labeled
        y_l_pred = model(X_l)

        # unlabeled
        # K times Data Augmentation
        y_u_tilde_pred_list: List[Tensor] = []
        y_u_pred_list: List[Tensor] = []
        for _ in range(K):
            _, X_u_tilde = pretext_generator(X=X_u, p_m=p_m, device=device)

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


def test(
    dataloader: DataLoader,
    model: SemiSLNetworks,
    loss_fn: nn.CrossEntropyLoss,
    device: str,
) -> None:
    size = len(dataloader.dataset)  # type: ignore
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X: Tensor = X.to(device)
            y: Tensor = y.to(device)

            y_logits = model(X)
            y_prob = softmax(input=y_logits, dim=1)

            test_loss += loss_fn(y_logits, y).item()
            correct += (y_prob.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
