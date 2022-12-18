from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import LongTensor, FloatTensor
from torch.utils.data import Dataset

from utils import one_hot_encoding, normalize


class UCIIncome(Dataset):
    def __init__(self, X: FloatTensor, y: LongTensor = None) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index) -> Tuple[FloatTensor, LongTensor] | FloatTensor:
        x = self.X[index]
        if self.y is None:
            return x
        else:
            y = self.y[index]
            return x, y

    @staticmethod
    def get_dataset(
        is_norm=True, train_size=0.8, labeled_size=0.1
    ) -> Tuple[FloatTensor, LongTensor, FloatTensor, FloatTensor, LongTensor]:
        target_column_name = "salary"

        # read csv
        df: pd.DataFrame = pd.read_csv("./data/uci_income/adult.csv", sep=",")

        # one-hot encoding
        df = one_hot_encoding(df, target_column_name)

        # normalization
        if is_norm:
            df = normalize(df)

        # data split
        train_data: np.ndarray
        test_data: np.ndarray
        train_l: np.ndarray
        train_u: np.ndarray
        train_data, test_data = train_test_split(df.values, train_size=train_size)
        train_l, train_u = train_test_split(train_data, train_size=labeled_size)

        # transform ndarray to tensor
        X_l_train = torch.from_numpy(train_l[:, :-1].astype(np.float32)).clone()
        y_train = torch.from_numpy(train_l[:, -1].astype(np.int64)).clone()
        X_u_train = torch.from_numpy(train_u[:, :-1].astype(np.float32)).clone()
        X_test = torch.from_numpy(test_data[:, :-1].astype(np.float32)).clone()
        y_test = torch.from_numpy(test_data[:, -1].astype(np.int64)).clone()

        return X_l_train, y_train, X_u_train, X_test, y_test
