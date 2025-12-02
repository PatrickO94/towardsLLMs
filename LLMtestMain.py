import torch
from torch.utils.data import DataLoader
from llmlib import KARPDataset, cfg
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
    """
    if torch.cuda.is_available():
        print("cuda device selected")
    else:
        print("cuda unavailable")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    kds = KARPDataset()
    train_ds, val_ds = kds.get_train_val_split(0.1)
    print(len(train_ds), len(val_ds))
    dl_train = DataLoader(train_ds, cfg.BATCH_SIZE, shuffle=True)
    print(next(iter(dl_train)))
    x, y = next(iter(dl_train))
    print(f"x-shape: {x.shape}, y-shape: {y.shape}")
