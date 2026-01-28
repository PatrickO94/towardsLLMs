import torch
from torch.utils.data import DataLoader
from llmlib import KARPDataset, cfg, BigramLM, estimate_loss, BigramBaseLM, DecoderAttentionLM
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
    """
    ########################---BIGRAM-BASE---################################
    m = BigramBaseLM(kds.vocab_size)
    # out, loss = m(x,y)
    # print(out.shape)
    # print(out)
    # print(loss)
    print(kds.tk.decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

    ######################---TRAIN-BIGRAM-BASE---##########################################
    optimizer = torch.optim.AdamW(m.parameters(), lr=cfg.LR)
    m.to(device)

    for epoch in range(cfg.EPOCHS):
        for x, y in dl_train:
            logits, loss = m.forward(x.to(device), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(epoch)
        out = estimate_loss(m, 200, dl_train, dl_val, device)
        print(out)
    print(loss.item())
    print(kds.tk.decode(m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
    
    #######################---BIGRAM-2-TRAIN---########################################
    m = BigramLM()
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(m.parameters(), lr=cfg.LR)
    m.to(device)

    for epoch in range(cfg.EPOCHS):
        for i, (x, y) in enumerate(dl_train):
            if i % cfg.EVAL_INTERVAL == 0:
                out = estimate_loss(m, cfg.EVAL_ITERS,dl_train, dl_val, device)
                print(out)
            logits, loss = m.forward(x.to(device), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(epoch)
        out = estimate_loss(m, cfg.EVAL_ITERS, dl_train, dl_val, device)
        print(out)
    print(kds.tk.decode(m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
    """
    #######################---BIGRAM-ATT-TRAIN---########################################
    m = DecoderAttentionLM()
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(m.parameters(), lr=cfg.LR)
    m.to(device)

    for epoch in range(cfg.EPOCHS):
        for i, (x, y) in enumerate(dl_train):
            if i % cfg.EVAL_INTERVAL == 0:
                out = estimate_loss(m, cfg.EVAL_ITERS, dl_train, dl_val, device)
                print(out)
            logits, loss = m.forward(x.to(device), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(epoch)
        out = estimate_loss(m, cfg.EVAL_ITERS, dl_train, dl_val, device)
        print(out)
    print(kds.tk.decode(m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=10000)[0].tolist()))