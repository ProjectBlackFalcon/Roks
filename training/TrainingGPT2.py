from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from DataGenerator import get_disk_posts
import os

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-2 tokenizer with byte-pair encoding
enc = GPT2Tokenizer.from_pretrained("gpt2")

# Pre-trained models with weights trained on WebText
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)
model.eval()

# Fetching data scraped from elsewhere
posts = get_disk_posts()
post = posts[0]

# Using the negative log likelihood loss function
criterion = nn.NLLLoss()

# Using the OpenAI Adam implementation and setting up the model weights with it
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = OpenAIAdam(optimizer_grouped_parameters, lr=0.9)

# Hyper-parameters
batch_size = 1
length = model.config.n_ctx // 2
temperature = 1


def train(model, target_tensor, criterion, optimizer):
    """
    Trains the model
    :param model: Actual GPT-2 model
    :param target_tensor: a list of tokens corresponding to the target document. Document should have been tokenified by
    the GPT-2 tokenizer
    :param criterion: The chosen loss function
    :param optimizer: The chosen optimizer
    :return: the loss value over this iteration
    """
    if len(target_tensor) > 512:
        raise Exception("Document contains more than 512 tokens.")

    # Initial parameters reset
    loss = 0
    optimizer.zero_grad()

    # Create a tensor and adapt it to the model accepted shape
    context = torch.tensor(target_tensor, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    # Iterate over every token
    for i in range(1, len(target_tensor)):
        # Define the part of the sentence which will be used as the context
        context_tokens = context[:i]
        # Obtain the prediction from the model
        logits, _ = model(context_tokens, past=None)
        # Temperature dictates the certainty with which the model makes predictions
        logits = logits[:, -1, :] / temperature
        # Gives a probability score to each predicted token; sum = 1
        log_probs = F.log_softmax(logits, dim=1)
        # Cast the target to tensor so it can be used in the criterion
        singular_target_tensor = torch.tensor([target_tensor[i]], device=device)

        # Calculate the loss using the probabilities compared to the expected output
        loss += criterion(log_probs, singular_target_tensor)

    loss.backward()
    optimizer.step()

    return loss.item() / len(target_tensor)


def train_iterations(model, target_tensors, criterion, optimizer, epochs, print_every=1, save_every=1):
    """
    Trains the model with every document
    :param model: Actual GPT-2 model
    :param target_tensors: list of all documents that will be fed to the model
    :param criterion: The chosen loss function
    :param optimizer: The chosen optimizer
    :param print_every: Every m iteration the average loss will be printed
    :return:
    """

    epoch = 0
    loss = 0
    iteration = 0

    checkpoints = sorted(os.listdir("checkpoints"))
    if len(checkpoints) > 0:
        checkpoint = checkpoints[len(checkpoints) - 1]
        checkpoint = torch.load("checkpoints/{}".format(checkpoint))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        iteration = checkpoint['iteration']
        print("File found. Loading at epoch {} and iterationc {}".format(epoch, iteration))
    else:
        print("No checkpoint found.")

    for e in trange(epoch, epochs):
        print_total_loss = loss
        # Iterate through every example and print the loss when necessary
        for iteration in range(iteration, len(target_tensors)):
            post = target_tensors[iteration]
            target_tensor = enc.encode(post[:10])
            loss = train(model, target_tensor, criterion, optimizer)
            print_total_loss += loss

            if iteration % print_every == 0:
                print("{}/{}, Loss: {}".format(iteration, len(target_tensors), print_total_loss / print_every))
                print_total_loss = 0

            if iteration % save_every == 0:
                torch.save({
                   'epoch': e,
                   'model_state_dict': model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
                   'loss': loss,
                   "iteration": iteration
                }, "checkpoints/gpt2_e-{}_i-{}.pt".format(e, iteration))


train_iterations(model, posts, criterion, optimizer, 2)
print("Training has ended.")
