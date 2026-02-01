import copy
from datetime import datetime
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from llmlib import KARPDataset, cfg, BigramLM, estimate_loss, BigramBaseLM, DecoderAttentionLM, OWT2Dataset
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


def save_checkpoint(model, name, epoch):
    model_state = copy.deepcopy(model.state_dict())
    current_date = datetime.now()
    formatted_date = current_date.strftime("_%Y_%m_%d_t_%H_%M")
    # Save original and quantized models
    torch.save(model_state, os.path.join(cfg.MDL_PATH, f"{name}{formatted_date}_E{epoch}.pth"))

def save_output(name, out_list):
    current_date = datetime.now()
    formatted_date = current_date.strftime("_%Y_%m_%d_t_%H_%M")
    with open(os.path.join(cfg.OUT_PATH, f'out{formatted_date}_Model_{name}.txt'), 'w') as file:
        file.writelines(item + '\n' for item in out_list)




if __name__ == "__main__":

    if torch.cuda.is_available():
        print("cuda device selected")
    else:
        print("cuda unavailable")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    #######################---DATA---###############################
    kds = KARPDataset()
    train_ds, val_ds = kds.get_train_val_split(0.1)
    print(len(train_ds), len(val_ds))
    dl_train = DataLoader(train_ds, cfg.BATCH_SIZE, shuffle=False)
    dl_val = DataLoader(val_ds, cfg.BATCH_SIZE, shuffle=False)
    # print(next(iter(dl_train)))
    # x, y = next(iter(dl_train))
    # print(f"x-shape: {x.shape}, y-shape: {y.shape}")
    
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
    """
    #######################---BIGRAM-ATT-TRAIN---########################################
    m = DecoderAttentionLM(vocab_size=65)
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(m.parameters(), lr=cfg.LR)
    m.to(device)

    for epoch in range(cfg.EPOCHS):
        for i, (x, y) in enumerate(dl_train):

            if i % cfg.EVAL_INTERVAL == 0:
                out = estimate_loss(m, cfg.EVAL_ITERS, dl_train, dl_val, device)
                print(f"train-loss: {out["train"]:.4f} | val-loss: {out["valid"]:.4f} | progress: {(i/len(dl_train))*100:.2f}% | Epoch {epoch + 1} of {cfg.EPOCHS}")
            logits, loss = m.forward(x.to(device), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        out = estimate_loss(m, cfg.EVAL_ITERS, dl_train, dl_val, device)
        print(f"train-loss: {out["train"]:.4f} | val-loss: {out["valid"]:.4f} | End of Epoch {epoch + 1} of {cfg.EPOCHS}")

    gen_batch = m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=10000)
    print(kds.tk.decode(gen_batch[0].tolist()))
    gens = [kds.tk.decode(b.tolist()) for b in gen_batch]
    save_checkpoint(m, "DecoderTransformerV1.0", cfg.EPOCHS)
    save_output("DecoderTransformerV1.0", gens)
    """
    #######################---ATT-TRAIN-OWT2--########################################
    print("initializing data...")
    owt2 = OWT2Dataset()
    # print(len(owt2))
    # print(owt2[0])
    # print(next(iter(DataLoader(owt2, batch_size=cfg.BATCH_SIZE, shuffle=False))))
    _, tmp = owt2.train_val_split(val_perc=0.25) # cropping ds to reduce training time
    train_ds, val_ds = tmp.train_val_split()
    del owt2, tmp, _
    print("length of train ds: ", len(train_ds))
    print("length of val ds: ", len(val_ds))
    dl_train = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    # print(next(iter(dl_train))[0])
    # print(next(iter(dl_val))[1])

    print("initializing model...")
    m = DecoderAttentionLM(vocab_size=train_ds.vocab_size)
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(m.parameters(), lr=cfg.LR)
    m.to(device)
    print("training starts...")
    for epoch in range(cfg.EPOCHS):
        for i, (x, y) in enumerate(dl_train):

            if i % cfg.EVAL_INTERVAL == 0:
                out = estimate_loss(m, cfg.EVAL_ITERS, dl_train, dl_val, device)
                print(f"train-loss: {out["train"]:.4f} | val-loss: {out["valid"]:.4f} | progress: {(i/len(dl_train))*100:.2f}% | Epoch {epoch + 1} of {cfg.EPOCHS}")
            logits, loss = m.forward(x.to(device), y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        out = estimate_loss(m, cfg.EVAL_ITERS, dl_train, dl_val, device)
        print(f"train-loss: {out["train"]:.4f} | val-loss: {out["valid"]:.4f} | End of Epoch {epoch + 1} of {cfg.EPOCHS}")

    gen_batch = m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=10000)
    print(train_ds.tokenizer.decode(gen_batch[0].tolist()))
    gens = [train_ds.tokenizer.decode(b.tolist()) for b in gen_batch]
    save_checkpoint(m, "DecoderTransformerV1.0_OWT2", cfg.EPOCHS)
    save_output("DecoderTransformerV1.0_OWT2", gens)
