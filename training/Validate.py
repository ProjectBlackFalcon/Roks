import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import GPT2Tokenizer

enc = GPT2Tokenizer.from_pretrained('gpt2')


def output_sentence(model, length=10, context_string=None, temperature=1, device='cpu'):
    context = enc.encode(context_string)
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(1, 1)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for _ in range(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1)
            output = torch.cat((output, prev), dim=1)

    return enc.decode(output.tolist()[0])
