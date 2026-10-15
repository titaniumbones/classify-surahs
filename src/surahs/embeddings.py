"""Verse embeddings from CAMeLBERT-CA, a BERT model trained on Classical Arabic.

An *embedding* is a list of 768 numbers the network produces for a verse;
verses the model "reads" as similar get similar lists. We average the
network's output over all the tokens in a verse (mean pooling) and then
train an ordinary logistic regression on top, exactly like the other
feature sets. Fine-tuning the whole network would be more powerful but much
slower, and with only 114 surahs it would overfit easily.

Needs the optional dependencies:  uv sync --extra transformers
The first run downloads the model (~450 MB) and takes several minutes on a
laptop CPU; the result is cached in data/cache/.

Caution: CAMeLBERT-CA's training corpus (OpenITI) contains a great deal of
classical Islamic literature, including tafsir and asbab al-nuzul works that
discuss which verses are Meccan or Medinan. The model may therefore carry
traces of the very tradition we are testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from surahs.data import DATA

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-ca"


def verse_embeddings(verses: pd.DataFrame, batch_size: int = 32, cache: bool = True) -> np.ndarray:
    path = DATA / "cache" / "camelbert_ca_verses.npy"
    if cache and path.exists():
        X = np.load(path)
        if len(X) == len(verses):
            return X
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    texts = verses["text"].tolist()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            out.append(((h * mask).sum(1) / mask.sum(1)).numpy())
    X = np.vstack(out).astype(np.float32)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, X)
    return X
