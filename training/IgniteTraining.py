from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam
import torch

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-2 tokenizer with byte-pair encoding
enc = GPT2Tokenizer.from_pretrained("gpt2")

# Pre-trained models with weights trained on WebText
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

train_loader, val_loader = get_data_loaders