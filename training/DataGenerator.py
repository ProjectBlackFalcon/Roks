import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
import numpy as np
import torch.nn.functional as F


def array_length(arr):
	return len(arr)


class PCPostsDataset(Dataset):
	"""
	Dataset object that returns examples of [tensor(post), tensor(firstComment)]
	The goal is to train a model to respond a comment to a post
	"""
	def __init__(self):
		print()

	def __len__(self):
		return 0

	def __getitem__(self, item):
		return 0


class LMPostsDataset(Dataset):
	"""
	Dataset object that returns examples of tensor(postDocument)
	This postDocument may be a post or a comment.
	The goal is to train a model to do next word prediction (LM)
	"""
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

	@staticmethod
	def collate(values):
		return torch.stack([F.pad(value, (len(values[-1]) - value.size(0), 0), "constant", 0) for value in values])


def get_data_loaders(train_batch_size,
                     val_batch_size,
                     device='cpu',
                     validation_split=0.2,
                     random_seed=0,
                     max_context_length=512):
	dataset = LMPostsDataset(device=device, max_context_length=max_context_length)
	dataset_size = len(dataset)

	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))

	np.random.seed(random_seed)
	np.random.shuffle(indices)

	train_indices, val_indices = indices[split:], indices[:split]
	train_sampler = SequentialSampler(train_indices)
	valid_sampler = SequentialSampler(val_indices)

	train_loader = DataLoader(dataset, collate_fn=LMPostsDataset.collate, batch_size=train_batch_size, sampler=train_sampler)
	val_loader = DataLoader(dataset, collate_fn=LMPostsDataset.collate, batch_size=val_batch_size, sampler=valid_sampler)

	return train_loader, val_loader
