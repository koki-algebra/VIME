import torch

from datasets import UCIIncome
from vime import SelfSLNetworks, SemiSLNetworks


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    X_l_train, y_train, X_u_train, X_test, y_test = UCIIncome.get_dataset(
        is_norm=True, train_size=0.8, labeled_size=0.1
    )
    train_labeled_dataset = UCIIncome(X=X_l_train, y=y_train)
    train_unlabeled_dataset = UCIIncome(X=X_u_train)
    test_dataset = UCIIncome(X=X_test, y=y_test)

    # number of features
    num_features = X_l_train.shape[1]
    dim_z = num_features * 2
    dim_y = 2

    # hyperparameters
    self_hyperparams = {
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 10,
        "p_m": 0.4,
        "alpha": 4.0,
    }
    semi_hyperparams = {
        "labeled_batch_size": 128,
        "unlabeled_batch_size": 128 * 9,
        "lr": 1e-3,
        "epochs": 20,
        "p_m": 0.4,
        "beta": 1.5,
        "K": 5,
    }

    # -------- Self Supvervised leaning --------
    # model
    self_model = SelfSLNetworks(dim_x=num_features, dim_z=dim_z)
    self_model.fit(
        dataset=train_unlabeled_dataset, hyperparams=self_hyperparams, device=device
    )

    # get trained encoder
    encoder = self_model.get_encoder()

    # save weights of the encoder
    torch.save(encoder.state_dict(), "encoder_weights.pth")

    # # -------- Semi-Supervised Learning --------
    # model
    semi_model = SemiSLNetworks(encoder=encoder, dim_z=dim_z, dim_y=2)
    semi_model.fit(
        labeled_dataset=train_labeled_dataset,
        unlabeled_dataset=train_unlabeled_dataset,
        test_dataset=test_dataset,
        hyperparams=semi_hyperparams,
        device=device,
    )
