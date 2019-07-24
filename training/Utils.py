import torch
import torch.nn.functional as F
from tqdm import trange


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    logits = logits[0]
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k and top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p and top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    logits.unsqueeze_(0)
    return logits


def sample_sequence(model, tokenizer, device, length=10, context=None, temperature=1.0):
    if context is not None:
        context = tokenizer.encode(context)
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0)
    else:
        context = torch.tensor([tokenizer.encoder["<|endoftext|>"]], device=device, dtype=torch.long).unsqueeze(0)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for _ in trange(length):
            logits, past = model(prev, past=past)
            logits = top_k_top_p_filtering(logits[:, -1, :] / temperature, top_p=0.9, top_k=40)
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1)
            output = torch.cat((output, prev), dim=1)

    output = output[0].tolist()
    output = tokenizer.decode(list(filter(lambda token: token != 50256, output)))
    return output
