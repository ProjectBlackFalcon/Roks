import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import numpy as np
import torch.nn.functional as F


def get_disk_posts(clean=False, irrelevant=False):
	with open("../data/{}posts.txt".format("clean_" if clean else "irrelevant_" if irrelevant else ""),
	          encoding="utf-8") as reader:
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
		input_batches.append(tokens[i:i + max_size])
		output_batches.append(tokens[i + max_size])

	return torch.tensor(input_batches), torch.tensor(output_batches)


def array_length(arr):
	return len(arr)


def collate(values):
	return torch.stack([F.pad(value, (len(values[-1]) - value.size(0), 0), "constant", 0) for value in values])


class PostsDataset(Dataset):
	def __init__(self, max_context_length=512, device='cpu'):
		with open("../data/tokenized_posts.txt") as file:
			file = file.read().split("\n")

		self.dataset_examples = []

		for item in file:
			item = list(map(int, filter(lambda i: len(i) > 0, item.split('|'))))
			while len(item) > 40:
				self.dataset_examples.append(item[:max_context_length])
				item = item[max_context_length:]

		self.dataset_examples.sort(key=array_length)

		self.max_context_length = max_context_length
		self.device = device

	def __len__(self):
		return len(self.dataset_examples)

	def __getitem__(self, item):
		return torch.tensor(self.dataset_examples[item], device=self.device)


def get_data_loaders(tokenizer,
                     train_batch_size,
                     val_batch_size,
                     device='cpu',
                     validation_split=0.2,
                     random_seed=0,
                     max_context_length=512):
	dataset = PostsDataset(device=device, max_context_length=max_context_length)
	dataset_size = len(dataset)

	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))

	np.random.seed(random_seed)
	np.random.shuffle(indices)

	train_indices, val_indices = indices[split:], indices[:split]
	train_sampler = SequentialSampler(train_indices)
	valid_sampler = SequentialSampler(val_indices)

	train_loader = DataLoader(dataset, collate_fn=collate, batch_size=train_batch_size, sampler=train_sampler)
	val_loader = DataLoader(dataset, collate_fn=collate, batch_size=val_batch_size, sampler=valid_sampler)

	return train_loader, val_loader
