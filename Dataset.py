from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch


def create_dataset():
    # Total number of samples
    data_len = 10000
    # Length of each sample
    input_dim = 10
    # Percentage of positive samples
    per = 0.7
    # (pos_x, pos_y, power)
    dataset = np.zeros((data_len, input_dim, 3))
    dataset[:, :, 2] = 1
    # labels = np.zeros(data_len)
    labels_2d = np.zeros((data_len, 4))
    # Tolerance
    tol = 0.1

    # Base speed
    v_x = 1
    v_y = 0
    v_p = 0.1

    for i in range(data_len):
        if random.random() < per:
            labels_2d[i,0] = 1
            for j in range(1, input_dim):
                dataset[i, j, 0] = dataset[i, j - 1, 0] + v_x * (1 + np.random.uniform(-1, 1) * tol)
                dataset[i, j, 1] = dataset[i, j - 1, 1] + np.random.uniform(-1, 1) * tol
                dataset[i, j, 2] = dataset[i, j - 1, 2] - v_p * (1 + np.random.uniform(-1, 1) * tol)
        else:
            err1 = random.randint(1, input_dim-1)
            err2 = random.randint(0, 2)
            for j in range(1, input_dim):
                dataset[i, j, 0] = dataset[i, j - 1, 0] + v_x * (1 + np.random.uniform(-1, 1) * tol)
                dataset[i, j, 1] = dataset[i, j - 1, 1] + np.random.uniform(-1, 1) * tol
                dataset[i, j, 2] = dataset[i, j - 1, 2] - v_p * (1 + np.random.uniform(-1, 1) * tol)
                if j == err1:
                    if err2 == 0:
                        dataset[i, j, 0] = dataset[i, j - 1, 0] + v_x * (1 + random.choice([-1, 1]) * (tol + np.random.uniform(0, 1) * (1-tol)))
                        labels_2d[i, 1] = 1
                    elif err2 == 1:
                        dataset[i, j, 1] = dataset[i, j - 1, 1] + random.choice([-1, 1]) * (tol + np.random.uniform(0, 1) * (1-tol))
                        labels_2d[i, 2] = 1
                    elif err2 == 2:
                        dataset[i, j, 2] = dataset[i, j - 1, 2] - v_p * (1 + random.choice([-1, 1]) * (tol + np.random.uniform(0, 1) * (1-tol)))
                        labels_2d[i,3] = 1

    dataset_ = torch.tensor(dataset, dtype=torch.float32)
    labels_ = torch.tensor(labels_2d, dtype=torch.float32)
    torch.save(dataset_, 'Dataset/data2.mat')
    torch.save(labels_, 'Dataset/label2.mat')
    print("...Create Finished...")


create_dataset()
# dataset = torch.load('Dataset/data1.mat')
# labels = torch.load('Dataset/label1.mat')
# print("...Load Finished...")
