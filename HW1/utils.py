import os
import random

import numpy as np
import torch


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_oct_data(bias=False, eq_test_dist=False, root="."):
    """
    Loads the preprocessed, featurized octmnist dataset, optionally adding a bias feature.
    """
    path = os.path.join(root, "octmnist.npz")
    data = np.load(path)
    train_X = data["train_images"].reshape([data["train_images"].shape[0], -1])/256
    dev_X = data["val_images"].reshape([data["val_images"].shape[0], -1])/256
    test_X = data["test_images"].reshape([data["test_images"].shape[0], -1])/256

    test_y = np.asarray(data["test_labels"]).squeeze()
    if not eq_test_dist:
        test_y_class0 = test_y[test_y == 0][0:182] #182
        test_X_class0 = test_X[test_y == 0][0:182] #182
        test_y_class1 = test_y[test_y == 1][0:55] #55
        test_X_class1 = test_X[test_y == 1][0:55] #55
        test_y_class2 = test_y[test_y == 2][0:42] #42
        test_X_class2 = test_X[test_y == 2][0:42] #42
        test_y_class3 = test_y[test_y == 3][0:250] #250
        test_X_class3 = test_X[test_y == 3][0:250] #250
        test_X = np.vstack((test_X_class0,
                             test_X_class1,
                             test_X_class2,
                             test_X_class3))
        test_y = np.hstack((test_y_class0,
                             test_y_class1,
                             test_y_class2,
                             test_y_class3))
    if bias:
        train_X = np.hstack((train_X, np.ones((train_X.shape[0], 1))))
        dev_X = np.hstack((dev_X, np.ones((dev_X.shape[0], 1))))
        test_X = np.hstack((test_X, np.ones((test_X.shape[0], 1))))
    return {"train": (train_X, np.asarray(data["train_labels"]).squeeze()),
            "dev": (dev_X, np.asarray(data["val_labels"]).squeeze()),
            "test": (test_X, test_y)}

class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        """
        data: the dict returned by utils.load_pneumonia_data
        """
        train_X, train_y = data["train"]
        dev_X, dev_y = data["dev"]
        test_X, test_y = data["test"]

        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.long)

        self.dev_X = torch.tensor(dev_X, dtype=torch.float32)
        self.dev_y = torch.tensor(dev_y, dtype=torch.long)

        self.test_X = torch.tensor(test_X, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
