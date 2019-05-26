from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from DataGenerator import get_disk_posts

enc = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

posts = get_disk_posts()
post = posts[0]

criterion = nn.NLLLoss()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = OpenAIAdam(optimizer_grouped_parameters, lr=0.9)

batch_size = 1
length = model.config.n_ctx // 2
temperature = 1


def train(model, target_tensor, criterion, optimizer):
    if len(target_tensor) > 512:
        raise Exception("Document contains more than 512 tokens.")

    loss = 0
    optimizer.zero_grad()

    for i in trange(1, len(target_tensor)):
        context_tokens = target_tensor[:i]
        context = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
        logits, past = model(context, past=None)
        logits = logits[:, -1, :] / temperature
        log_probs = F.log_softmax(logits, dim=1)
        singular_target_tensor = torch.tensor([target_tensor[i]])

        loss += criterion(log_probs, singular_target_tensor)

    print("Loss backward...")
    loss.backward()
    print("Optimizer step + zero_grad ...")
    optimizer.step()

    return loss.item() / len(target_tensor)


train(model, enc.encode(post)[:100], criterion, optimizer)
print("Training has ended.")
# print(len(enc.encode(post)[:150]))