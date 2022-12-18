from typing import Tuple, Dict
import torch.nn as nn
from torch import Tensor, sigmoid
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, Adam

from .pretext import pretext_generator


class SelfSLNetworks(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, is_norm=True) -> None:
        super().__init__()
        self.is_norm = is_norm

        # encoder e: X -> Z
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(dim_x),
            nn.Linear(dim_x, dim_x * 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_x * 2),
            nn.Linear(dim_x * 2, dim_x * 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_x * 2),
            nn.Linear(dim_x * 2, dim_z),
        )

        # mask vector estimator s_m: Z -> {0,1}^d
        self.mask_estimator = nn.Sequential(
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_z),
            nn.ReLU(),
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_x),
        )

        # feature vector estimator s_r: Z -> X
        self.feature_estimator = nn.Sequential(
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_z),
            nn.ReLU(),
            nn.BatchNorm1d(dim_z),
            nn.Linear(dim_z, dim_x),
        )

    def forward(self, X_tilde: Tensor) -> Tuple[Tensor, Tensor]:
        # encode collapsed feature matrix
        Z: Tensor = self.encoder(X_tilde)
        # mask prediction
        M_pred: Tensor = sigmoid(self.mask_estimator(Z))
        # feature prediction
        X_pred: Tensor = self.feature_estimator(Z)

        # is normalization
        if self.is_norm:
            X_pred = sigmoid(X_pred)

        return M_pred, X_pred

    def fit(
        self, dataset: Dataset, hyperparams: Dict[str, float | int], device: str
    ) -> None:
        # hyperparams
        batch_size: int = hyperparams["batch_size"]
        learning_rate: float = hyperparams["lr"]
        epochs: int = hyperparams["epochs"]
        p_m: float = hyperparams["p_m"]
        alpha: float = hyperparams["alpha"]

        if batch_size < 0:
            raise ValueError("batch size must be greater than 0")
        if learning_rate < 0.0:
            raise ValueError("learning rate must be greater than 0.0")
        if epochs < 0:
            raise ValueError("epochs must be greater than 0")
        if p_m > 1.0 or p_m < 0.0:
            raise ValueError(
                "p_m must be greater than 0.0 and less than 1.0, e.g. p_m ∈ (0.0, 1.0)"
            )
        if alpha < 0.0:
            raise ValueError("alpha must be greater than 0.0")

        # model
        model = self.to(device)
        # loss function
        loss_fn = SelfSLLoss(alpha)
        # optimizer
        optimizer = Adam(params=model.parameters(), lr=learning_rate)

        # data loader
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # training
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(
                dataloader=dataloader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                p_m=p_m,
            )
        print("Done!")

    def get_encoder(self) -> nn.Module:
        return self.encoder


class SelfSLLoss(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self, M_pred: Tensor, M_target: Tensor, X_pred: Tensor, X_target: Tensor
    ) -> float:
        mask_loss = nn.BCELoss()
        feature_loss = nn.MSELoss()
        return mask_loss(M_pred, M_target) + self.alpha * feature_loss(X_pred, X_target)


def train(
    dataloader: DataLoader,
    model: SelfSLNetworks,
    loss_fn: SelfSLLoss,
    optimizer: Optimizer,
    device: str,
    p_m=0.2,
) -> None:
    size = len(dataloader.dataset)  # type: ignore
    model.train()
    for batch, X in enumerate(dataloader):
        X: Tensor = X.to(device)
        # pretext generate
        M, X_tilde = pretext_generator(X, p_m, device)

        # compute prediction and loss
        M_pred, X_pred = model(X_tilde)

        loss = loss_fn(M_pred, M, X_pred, X)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
