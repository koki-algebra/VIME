from torch.utils.data import DataLoader

from datasets import UCIIncome

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