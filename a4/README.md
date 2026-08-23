# Introduction

## File Guide

| File | Purpose |
| ---- | ---- |
| run.py | Main training script: loads data, builds the model, trains, validates, and writes logs |
| run.sh | One-click training: wraps all the arguments and calls run.py |
| nmt_model.py | Model core: bidirectional LSTM encoder + attention decoder (the part you implement in the assignment) |
| model_embeddings.py | Embedding layers: source/target embeddings + CNN layer |
| vocab.py | Vocab utilities (Vocab class): sentence ↔ word-id conversion; can also train a new vocab |
| utils.py | Shared helpers: padding, batching, model save/load, etc. |
| loss.py | Reads TensorBoard logs and plots the loss curve into loss_curve.png |
| sanity_check.py | Small-scale sanity check (En-Es data) to verify forward/backward pass is correct |
| zh_en_data/ | Chinese-English parallel corpus (train + dev splits) |
| sanity_check_en_es_data/ | Small En-Es dataset for the sanity check |
| src.model / tgt.model | SentencePiece tokenizer models for Chinese/English (outputs of vocab.py) |
| src.vocab | Source-language SentencePiece vocab (byproduct of vocab.py) |
| vocab.json | The actual vocab used in training: 30001 Chinese words + 8001 English words |
| outputs/ | Where trained model weights are saved (model.bin) |
| runs/ | TensorBoard logs: nmt = GPU training, nmt_local = CPU training |

## Parameter Distribution

embed_size=1024, hidden_size=768, dropout_rate=0.3

| Parameter Block | Parameters | Description |
| ---- | ---- | ---- |
| model_embeddings.source.weight | 30,721,024 (30.7M) | Source language (zh) word embedding: 30001 × 1024 |
| model_embeddings.target.weight | 8,193,024 (8.2M) | Target language (en) word embedding: 8001 × 1024 |
| target_vocab_projection.weight | 6,144,768 (6.1M) | Output projection layer: 8001 × 768 |
| decoder.weight_ih | 5,505,024 (5.5M) | Decoder LSTM input weights (including 4×768×1792 for input feeding) |
| encoder.weight_ih_l0(_reverse) ×2 | 3,145,728 (3.1M) each | Bidirectional encoder LSTM forward/backward input weights |
| encoder.weight_hh_l0(_reverse) ×2 | 2,359,296 (2.4M) each | Encoder LSTM hidden‑to‑hidden weights |
| decoder.weight_hh | 2,359,296 (2.4M) | Decoder LSTM hidden‑to‑hidden weights |
| post_embed_cnn.weight | 2,097,152 (2.1M) | Post‑embedding CNN layer |
| combined_output_projection.weight | 1,769,472 (1.8M) | Attention concatenated output projection: 2304 × 768 |
| h_projection / c_projection / att_projection | 1,179,648 (1.2M) each | Initial hidden state / attention projection: 768×768×2 |
| Others (biases, etc.) | Minor | — |
