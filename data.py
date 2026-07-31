import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .model import ENCODER_NAME


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_label_map(path):
    with open(path) as f:
        return json.load(f)


def load_label_freq(path):
    with open(path) as f:
        return json.load(f)


class MeshDataset(Dataset):
    def __init__(self, jsonl_path, label_map, tokenizer, max_len=256):
        self.records = load_jsonl(jsonl_path)
        self.label_map = label_map
        self.num_labels = len(label_map)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec["text"]
        label_ids = [self.label_map[u] for u in rec["labels"] if u in self.label_map]

        y = torch.zeros(self.num_labels, dtype=torch.float32)
        y[label_ids] = 1.0

        enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
                              padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": y,
            "label_ids": label_ids,
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    label_ids = [b["label_ids"] for b in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "label_ids": label_ids}


def build_tokenizer():
    return AutoTokenizer.from_pretrained(ENCODER_NAME)


def build_freq_tensor(label_map, label_freq, device):
    inv_label_map = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)
    freq = torch.zeros(num_labels, dtype=torch.float32)
    for i in range(num_labels):
        freq[i] = float(label_freq.get(inv_label_map[i], 0))
    return freq.to(device)
