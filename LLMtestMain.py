import torch

if torch.cuda.is_available():
    print("cuda device selected")
else:
    print("cuda unavailable")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")