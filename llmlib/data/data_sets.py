from torch.utils.data import Dataset
import torch

from llmlib.config import cfg
from llmlib.data import KARP_DATA
from llmlib.data.util import CharLvlTokeniser
from copy import copy, deepcopy


class KARPDataset(Dataset):
    def __init__(self):
        self.text = ""
        self._get_karp_data()
        self.chars = ""
        self.vocab_size = 0
        self._get_vocab_size()
        self.tk = CharLvlTokeniser(self.chars)
        self.data = self._transmute_text()



    def __str__(self):
        return ("Example text, first 500 characters:\n"+
                self.text[:500]+"\n"+
                "---------------------------------------------------------\n"+
                f"length of text: {len(self.text)}\n"+
                f"vocab size: {self.vocab_size}\n"+
                "vocabulary:\n"+
                (''.join(self.chars))+"\n"+
                "First 500 tokens of data tensor:\n"+
                str(self.data[:500])+"\n")

    def get_train_val_split(self, val_percent):
        idx = int(len(self.data) * (1-val_percent))
        train_set = copy(self)
        train_set.data = self.data[:idx]
        val_set = copy(self)
        val_set.data = self.data[idx:]
        return train_set, val_set

    def _get_karp_data(self):
        with open(KARP_DATA, 'r', encoding='utf-8') as f:
            text = f.read()
        self.text = text

    def _get_vocab_size(self):
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

    # return torch.tensors of tokenised data text
    def _transmute_text(self):
        return torch.tensor(self.tk.encode(self.text), dtype=torch.long)

    def print_example(self, num_chars):
        assert num_chars < len(self.text), "not enough chars in data text"
        print(self.text[:num_chars])

    def __len__(self):
        return len(self.data) - cfg.CONTEXT_LEN

    def __getitem__(self, idx):
        x = self.data[idx:idx+cfg.CONTEXT_LEN]
        y = self.data[idx+1:idx+cfg.CONTEXT_LEN+1]
        return x, y