import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_disk_posts(clean=False, irrelevant=False):
    with open("../data/{}posts.txt".format("clean_" if clean else "irrelevant_" if irrelevant else ""), encoding="utf-8") as reader:
        return reader.read()


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
    def __init__(self, tokenizer, cache=None, max_context_length=512, device='cpu', step=None):
        if cache:
            with open('../data/{}.txt'.format(cache), 'r', encoding="utf-8") as file:
                self.posts = list(map(int, file.read().split("|")))
        else:
            posts = get_disk_posts(clean=True)
            tokenized_posts = tokenizer.tokenize(posts)
            self.posts = tokenizer.convert_tokens_to_ids(tokenized_posts)

        self.max_context_length = max_context_length
        self.device = device
        self.step = max_context_length/2 if step is None else step

    def save_to_cache(self, cache):
        with open('../data/{}.txt'.format(cache), 'w+', encoding="utf-8") as file:
            file.write(str('|'.join(str(post) for post in self.posts)))

    def __len__(self):
        print(len(self.posts))
        return len(self.posts) - self.max_context_length - 1

    def __getitem__(self, item):
        return (
            torch.tensor(self.posts[item:item+self.max_context_length]),
            torch.tensor(self.posts[item+1:item+self.max_context_length+1])
        )


def get_data_loaders(tokenizer, train_batch_size, val_batch_size, validation_split=0.2, random_seed=0):
    dataset = PostsDataset(tokenizer, cache="dataset_cache")
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=val_batch_size, sampler=valid_sampler)

    print("Returning data")

    return train_loader, val_loader
