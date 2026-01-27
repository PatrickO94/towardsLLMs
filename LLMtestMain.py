import torch
from torch.utils.data import DataLoader
from llmlib import KARPDataset, cfg, BigramLM
import os
from time import sleep

"""
def clear_line():
    columns, _ = os.get_terminal_size()
    print(' ' * (columns - 1), end='\r')

for x in range(100):
    clear_line()
    print(f"Progress: {x}%", end='\r')
    sleep(0.1)   
"""
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



if __name__ == "__main__":

    if torch.cuda.is_available():
        print("cuda device selected")
    else:
        print("cuda unavailable")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #######################---DATA---###############################
    kds = KARPDataset()
    train_ds, val_ds = kds.get_train_val_split(0.1)
    print(len(train_ds), len(val_ds))
    dl_train = DataLoader(train_ds, cfg.BATCH_SIZE, shuffle=False)
    dl_val = DataLoader(val_ds, cfg.BATCH_SIZE, shuffle=False)
    # print(next(iter(dl_train)))
    # x, y = next(iter(dl_train))
    # print(f"x-shape: {x.shape}, y-shape: {y.shape}")

    ########################---BIGRAM---################################
    m = BigramLM(kds.vocab_size)
    # out, loss = m(x,y)
    # print(out.shape)
    # print(out)
    # print(loss)
    print(kds.tk.decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

    ######################---TRAIN-BIGRAM---##########################################
    optimizer = torch.optim.AdamW(m.parameters(), lr=cfg.LR)
    m.to(device)

    for epoch in range(cfg.EPOCHS):
        for x, y in dl_train:
            logits, loss = m.forward(x.to(device), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(epoch)
        out = estimate_loss(m, 50, dl_train, dl_val, device)
        print(out)
    print(loss.item())
    print(kds.tk.decode(m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))

