import json
import torch
from llmlib.config import cfg
from copy import copy, deepcopy
from datasets import load_dataset
from llmlib.data import KARP_DATA
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer
from llmlib.data.util import CharLvlTokeniser
import random


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


class OWT2Dataset(Dataset):
    def __init__(self):
        self.hf_dataset = load_dataset(
            "Geralt-Targaryen/openwebtext2",
            split="train",
            streaming=False,
            download_mode="reuse_dataset_if_exists",
            cache_dir="/home/rick/PycharmProjects/towardsLLMs/llmlib/data/data"
        )
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = self.tokenizer.vocab_size  # 50257

    def train_val_split(self, val_perc=0.01):
        split = self.hf_dataset.train_test_split(test_size=val_perc)
        train = deepcopy(self)
        val = deepcopy(self)
        train.hf_dataset = split["train"]
        val.hf_dataset = split["test"]
        return train, val

    def __len__(self):
        return len(self.hf_dataset) - cfg.CONTEXT_LEN

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        txt = sample["text"]
        encoding = self.tokenizer(text=txt, max_length=cfg.CONTEXT_LEN+1, return_tensors="pt", return_attention_mask=False, padding='max_length', truncation=True)
        x = encoding["input_ids"][0][0:cfg.CONTEXT_LEN]
        y = encoding["input_ids"][0][1:cfg.CONTEXT_LEN+1]
        # print(x.shape, y.shape)
        return x, y

"""
class OWT2Dataset(Dataset):
    
    @article{pile,
    title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
    author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster,
    Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
    journal={arXiv preprint arXiv:2101.00027},
    year={2020}}

    import glob
    import os
    import math

    import tqdm

    from utils.archiver import Reader

    document_count = 0
    total_text_size = 0
    dataset_directory = "PATH_TO_FILES"
    files = glob.glob(os.path.join(dataset_directory, "*jsonl.zst"))
    for file_path in tqdm.tqdm(files, dynamic_ncols=True):
        reader = Reader()
        for document, metadata in reader.read_jsonl(file_path, get_meta=True):
            document_count += 1
            total_text_size += len(document)

    billion = math.pow(10, 9)
    print(f"Total Document Count: {document_count:,}")
    print(f"Total Uncompressed Text Size: {(total_text_size / billion):.2f} GB")
    
    def __init__(self, file_path):
        self.lines = []
        with open(file_path, 'r') as f:  # or handle multiple files/zst decompression
            for line in f:
                data = json.loads(line)
                self.lines.append(data['text'])  # adjust field if needed

    def __len__(self): return len(self.lines)
    def __getitem__(self, idx): return self.lines[idx]
# Then: dataset = OWT2Dataset('path/to/extracted/file.jsonl')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    def __init__(self):
        # Load the HF dataset (cached locally, no re-download)
        self.hf_dataset = load_dataset(
            "Geralt-Targaryen/openwebtext2",
            split="train",
            download_mode="reuse_dataset_if_exists",
            cache_dir="/home/rick/PycharmProjects/towardsLLMs/llmlib/data/data"
        )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = self.tokenizer.vocab_size  # 50257

        # Pre-tokenize everything once into a giant flat tensor (like your KARPDataset)
        # This may take 10-20 minutes first time, but then it's cached in RAM and blazing fast
        print("Tokenizing OpenWebText2 dataset... (this runs once and is cached in RAM)")
        tokenized = self.hf_dataset.map(
            lambda x: self.tokenizer(x["text"], truncation=False)["input_ids"],
            batched=True,
            num_proc=8,  # adjust based on your CPU cores
            remove_columns=self.hf_dataset.column_names
        )

        # Flatten all documents into one giant list of tokens
        self.data = torch.tensor([token for doc in tokenized for token in doc], dtype=torch.long)

        print(f"OWT2 loaded! Total tokens: {len(self.data):,}")
        print(f"Vocab size: {self.vocab_size}")
        print(f"Example text (first 500 chars after detokenization):\n{self._detokenize_first_500()}\n")

    def _detokenize_first_500(self):
        # Helper for nice __str__
        sample_tokens = self.data[:1000]  # take more to ensure we get clean text
        text = self.tokenizer.decode(sample_tokens, clean_up_tokenization_spaces=True)
        return text[:500] + ("..." if len(text) > 500 else "")

    def __str__(self):
        example_text = self._detokenize_first_500()
        return (f"Example text, first 500 characters:\n{example_text}\n"
                "---------------------------------------------------------\n"
                f"length of text (in tokens): {len(self.data):,}\n"
                f"vocab size: {self.vocab_size}\n"
                "First 500 tokens of data tensor:\n"
                f"{self.data[:500]}\n")

    def get_train_val_split(self, val_percent=0.01):  # 1% val is plenty for OWT2
        idx = int(len(self.data) * (1 - val_percent))
        train_set = copy.deepcopy(self)
        val_set = copy.deepcopy(self)
        train_set.data = self.data[:idx]
        val_set.data = self.data[idx:]
        return train_set, val_set

    def print_example(self, num_chars=500):
        text = self.tokenizer.decode(self.data[:1000], clean_up_tokenization_spaces=True)
        print(text[:num_chars])

    def __len__(self):
        return len(self.data) - cfg.CONTEXT_LEN

    def __getitem__(self, idx):
        x = self.data[idx:idx + cfg.CONTEXT_LEN]
        y = self.data[idx + 1:idx + cfg.CONTEXT_LEN + 1]
        return x, y
"""





