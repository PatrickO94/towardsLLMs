import torch

# check googles sentencePiece tokeniser for a more efficient but more complicated one.
class CharLvlTokeniser:
    def __init__(self, chars):
        self.str_to_int = {ch:i for i, ch in enumerate(chars)}
        self.int_to_str = {i:ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [self.str_to_int[c] for c in s]
        self.decode = lambda l: ''.join([self.int_to_str[i] for i in l])

@torch.no_grad()
def estimate_loss(model, eval_iters, dl_train, dl_val, device):
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        itt = iter(dl_train)
        itv = iter(dl_val)
        for k in range(eval_iters):
            try:
                x, y = next(itt) if split == 'train' else next(itv)
            except StopIteration:
                # Restart the iterator if the DataLoader is exhausted
                itt = iter(dl_train)
                itv = iter(dl_val)
                x, y = next(itt) if split == 'train' else next(itv)
            logits, loss = model(x.to(device),y.to(device))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out