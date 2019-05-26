from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
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
context_tokens = enc.encode(post)
criterion = nn.NLLLoss()
batch_size = 1
length = model.config.n_ctx // 2
temperature = 1

context = torch.tensor(context_tokens, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
prev = context
output = context
past = None

for _ in trange(length):
    logits, past = model(prev, past=past)
    logits = logits[:, -1, :] / temperature
    log_probs = F.softmax(logits, dim=1)
    prev = torch.multinomial(log_probs, num_samples=1)
    output = torch.cat((output, prev), dim=1)


out = output[:, len(context_tokens):].tolist()
text = enc.decode(out[0])
print(text)