import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_disk_posts(clean=False, irrelevant=False):
    with open("../data/{}posts.txt".format("clean_" if clean else "irrelevant_" if irrelevant else ""), encoding="utf-8") as reader:
        data = reader.read().split("\n")
    return data


def save_data(data, data_name, append=False):
    with open('../data/{}.txt'.format(data_name), 'a+' if append else 'w+', encoding="utf-8") as file:
        file.write(data)
        print("wrote", data, "to file")


def read_tokens(data_name):
    with open('../data/{}.txt'.format(data_name), 'r', encoding="utf-8") as file:
        token_string = file.read()
        tokens = token_string.split("\n")[:-1]
        return [list(map(int, single.split("|")[:-1])) for single in tokens]


def get_tensors_from_tokens(tokens, max_size=512):
    input_batches = []
    output_batches = []

    for i in range(len(tokens) - max_size):
        input_batches.append(tokens[i:i+max_size])
        output_batches.append(tokens[i+max_size])

    return torch.tensor(input_batches), torch.tensor(output_batches)


class PostsDataset(Dataset):
    def __init__(self):
        self.posts = get_disk_posts(clean=True)

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, item):
        return self.posts[item]


def get_data_loaders(train_batch_size, val_batch_size, validation_split=0.2, random_seed=0):
    dataset = PostsDataset()
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=val_batch_size, sampler=valid_sampler, shuffle=True)

    return train_loader, val_loader
