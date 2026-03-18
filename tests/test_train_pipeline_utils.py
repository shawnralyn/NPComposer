import types
from pathlib import Path

import pandas as pd
import pytest
import torch

train_mod = pytest.importorskip("src.training.train")


class _DummyTokenizer:
    def __init__(self):
        self.additional_special_tokens = []
        self._tok_to_id = {}
        self.eos_token = "<eos>"
        self.pad_token = None

    def add_special_tokens(self, special_tokens_dict):
        toks = special_tokens_dict.get("additional_special_tokens", [])
        n_before = len(self.additional_special_tokens)
        for t in toks:
            if t not in self.additional_special_tokens:
                self.additional_special_tokens.append(t)
        return len(self.additional_special_tokens) - n_before

    def __len__(self):
        return 100 + len(self.additional_special_tokens)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        out = []
        for t in tokens:
            if t not in self._tok_to_id:
                self._tok_to_id[t] = 1000 + len(self._tok_to_id)
            out.append(self._tok_to_id[t])
        return out

    def __call__(self, text, add_special_tokens=False, **kwargs):
        # Minimal tokenization: split on spaces; ensure deterministic length.
        ids = list(range(len(str(text).split())))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _DummyModel:
    def __init__(self):
        self._resized_to = None

    def resize_token_embeddings(self, n):
        self._resized_to = n


def test_build_special_class_tokens(tmp_path):
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN"],
            "pathway": ["A", "B"],
            "superclass": ["S1", "S1"],
            "is_glycoside": [0, 1],
            "aromatic": [0, 2],
            "qed_bin": ["0<=qed<0.1", "0.1<=qed<0.2"],
            "sa_bin": ["1<=sa<2", "2<=sa<3"],
        }
    )
    p = tmp_path / "train.csv"
    df.to_csv(p, index=False)

    toks = train_mod.build_special_class_tokens(
        str(p),
        pathway_col="pathway",
        superclass_col="superclass",
        is_glycoside_col="is_glycoside",
        num_aromatic_rings_col="aromatic",
        qed_bin_col="qed_bin",
        sa_bin_col="sa_bin",
    )

    assert "additional_special_tokens" in toks
    tok_list = toks["additional_special_tokens"]
    assert "<pathway:A>" in tok_list
    assert "<pathway:B>" in tok_list
    assert "<superclass:S1>" in tok_list


def test_update_tokenizer_with_special_tokens_resizes_embeddings():
    model = _DummyModel()
    tok = _DummyTokenizer()

    special_tokens_dict = {"additional_special_tokens": ["<x:1>", "<y:2>"]}
    model, tok = train_mod.update_tokenizer_with_special_tokens(model, tok, special_tokens_dict)

    assert "<x:1>" in tok.additional_special_tokens
    assert model._resized_to == len(tok)


def test_dataframe_to_tokenized_dataset_filters_overlong_rows():
    tok = _DummyTokenizer()
    df = pd.DataFrame(
        {
            "sm": ["CCO", "CCO"],
            "path": ["A", "A"],
            "sup": ["S", "S"],
            "gly": [0, 0],
            "ar": [0, 0],
            "q": ["0<=qed<0.1", "0<=qed<0.1"],
            "sa": ["1<=sa<2", "1<=sa<2"],
        }
    )
    # Force tokenization length = number of space-separated tokens in train_text.
    # This will be constant here, so use filter_len=1 to drop all rows.
    ds = train_mod.dataframe_to_tokenized_dataset(
        df,
        tokenizer=tok,
        smiles_col="sm",
        pathway_col="path",
        superclass_col="sup",
        is_glycoside_col="gly",
        num_aromatic_rings_col="ar",
        qed_bin_col="q",
        sa_bin_col="sa",
        max_len=128,
        filter_len=1,
    )
    assert len(ds) == 0


def test_conditioning_dropout_collator_masks_special_tokens(monkeypatch):
    tok = _DummyTokenizer()
    tok.additional_special_tokens = ["<a:1>", "<b:2>"]
    special_ids = tok.convert_tokens_to_ids(tok.additional_special_tokens)

    # Monkeypatch the padding collator to avoid depending on transformers internals.
    def _fake_pad(batch):
        # assume all same length for simplicity
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        for x in batch:
            ids = x["input_ids"] + [0] * (max_len - len(x["input_ids"]))
            mask = x["attention_mask"] + [0] * (max_len - len(x["attention_mask"]))
            input_ids.append(ids)
            attention_mask.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    monkeypatch.setattr(train_mod, "DataCollatorWithPadding", lambda _tok: _fake_pad)

    collate = train_mod.conditioning_dropout_collator(
        tokenizer=tok,
        special_token_dropout_prob=0.0,
        drop_all_special_tokens_prob=0.0,
    )

    batch = [
        {"input_ids": [special_ids[0], 5, 6], "attention_mask": [1, 1, 1]},
        {"input_ids": [7, special_ids[1]], "attention_mask": [1, 1]},
    ]

    out = collate(batch)
    assert "labels" in out

    labels = out["labels"].tolist()
    inp = out["input_ids"].tolist()

    # Special token positions should be masked to -100
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if inp[i][j] in special_ids:
                assert labels[i][j] == -100
