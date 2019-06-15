from DataGenerator import get_data_loaders
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam

import torch
from torch.nn.parallel import DistributedDataParallel
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-2 tokenizer with byte-pair encoding
enc = GPT2Tokenizer.from_pretrained("gpt2")

# Pre-trained models with weights trained on WebText
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

train_loader, val_loader = get_data_loaders(enc, train_batch_size=16, val_batch_size=32)

optimizer = OpenAIAdam(model.parameters(), lr=6.25e-5)


def update(engine, batch):
    model.train()
    batch = torch.stack(batch).to(device)
    loss = model.forward(batch, lm_labels=batch)
    print(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

trainer = Engine(update)
trainer.run(train_loader, max_epochs=3)