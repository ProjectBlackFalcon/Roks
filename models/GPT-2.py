from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
from tqdm import trange

enc = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# raw_text = input("Prompt:")
raw_text = "My name is "
context_tokens = enc.encode(raw_text)
generated = 0


def sample_sequence(model, length=-1, batch_size=1, context=None, temperature=1):
    model.eval()
    context = torch.tensor(context, device='cpu', dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for _ in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1)
            output = torch.cat((output, prev), dim=1)

    return output


out = sample_sequence(
    model=model,
    context=context_tokens,
    length=model.config.n_ctx // 2
)

out = out[:, len(context_tokens):].tolist()
text = enc.decode(out[0])

print(f"{'=' * 40} SAMPLE {str(generated)} {'=' * 40}")
print(raw_text + text)