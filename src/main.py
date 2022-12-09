import torch
from torch.utils.data import DataLoader

from datasets import UCIIncome
from vime import SelfSLNetworks, SelfSLLoss, self_train


if __name__ == "__main__":
    # dataset
    X_l_train, y_train, X_u_train, X_test, y_test = UCIIncome.get_dataset(is_norm=True, train_size=0.8, labeled_size=0.1)
    train_labeled_dataset   = UCIIncome(X=X_l_train, y=y_train)
    train_unlabeled_dataset = UCIIncome(X=X_u_train)
    test_dataset            = UCIIncome(X=X_test, y=y_test)

    # dataloader
    train_labeled_dataloader = DataLoader(
        dataset=train_labeled_dataset,
        batch_size=64
    )
    train_unlabeled_dataloader = DataLoader(
        dataset=train_unlabeled_dataset,
        batch_size=256
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64
    )

    print("Number of each data")
    print(f"train labeled   : {len(train_labeled_dataset)}")
    print(f"train unlabeled : {len(train_unlabeled_dataset)}")
    print(f"test            : {len(test_dataset)}")
    print()
    print("Number of each mini-batch")
    print(f"train labeled   : {len(train_labeled_dataloader)}")
    print(f"train unlabeled : {len(train_unlabeled_dataloader)}")
    print(f"test            : {len(test_dataloader)}")


    # number of features
    num_features = X_l_train.shape[1]

    # hyperparameters
    learning_rate = 1e-3
    epochs = 20
    p_m = 0.2
    alpha = 3.0
    beta = 3.0


    # -------- Self Supvervised leaning --------
    # model
    self_model = SelfSLNetworks(dim_x=num_features, dim_z=num_features*2)

    # loss function
    loss_fn = SelfSLLoss(alpha)

    # optimizer
    optimizer = torch.optim.Adam(params=self_model.parameters(), lr=learning_rate)

    # training
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        self_train(
            dataloader=train_unlabeled_dataloader,
            model=self_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            p_m=p_m
        )
    print("Done!")

    # get trained encoder
    encoder = self_model.get_encoder()

    # save weights of the encoder
    torch.save(encoder.state_dict(), "encoder_weights.pth")
