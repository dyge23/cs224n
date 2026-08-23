"""Evaluate the trained TinyStories Transformer.

Usage (run from the a3 directory):
    python evaluate.py                        # reads ./model_tiny.pt by default
    python evaluate.py --weights your_weights.pt
    python evaluate.py --compare-gpt2         # also compare against real GPT-2 (downloads weights)

What it evaluates:
    1. Quantitative: test loss and perplexity (= exp(loss)) on held-out chunks
    2. Qualitative: prompt with the beginning of a real story and print the continuation
"""

import argparse
import math
import random

import torch
from transformers import AutoTokenizer

from model_solution import Transformer, ModelConfig

# Must exactly match the config in train.py, otherwise loading weights will fail
# with a key mismatch error.
TINY_CONFIG = ModelConfig(
    d_model=33,
    n_heads=3,
    n_layers=3,
    context_length=512,
    vocab_size=50257,
)

DATASET_PATH = "./datasets/tinystories_10pct_chunk_size_512.pt"


def load_test_batches(chunk_size: int, num_batches: int, batch_size: int, device):
    """Hold out the last 10% of chunks from the cached TinyStories data as test set.

    Note: these chunks come from the same 1% slice as the training data, so this
    only measures in-distribution fit. To measure true generalization, chunk the
    untouched 99% of TinyStories the same way as get_chunked_tinystories in train.py.
    """
    dataset = torch.load(DATASET_PATH)
    n = dataset.shape[0]
    test = dataset[int(n * 0.9):]
    batches = []
    for i in range(0, len(test), batch_size):
        if len(batches) >= num_batches:
            break
        batches.append(test[i:i + batch_size].to(device))
    return batches


@torch.no_grad()
def evaluate_loss(model, batches):
    """Return (mean loss, perplexity). Perplexity = exp(loss), lower is better."""
    model.eval()
    total_loss, n = 0.0, 0
    for batch in batches:
        loss = model.get_loss_on_batch(batch)
        total_loss += loss.item() * batch.shape[0]
        n += batch.shape[0]
    avg_loss = total_loss / n
    return avg_loss, math.exp(avg_loss)


@torch.no_grad()
def generate_samples(model, tokenizer, device, num_samples=3, num_new_tokens=64):
    """Use the first 16 tokens of real stories from the test set as prompts and continue them."""
    dataset = torch.load(DATASET_PATH)
    test = dataset[int(dataset.shape[0] * 0.9):]
    rng = random.Random(0)
    model.eval()
    for _ in range(num_samples):
        chunk = test[rng.randrange(len(test))].to(device)
        prompt = chunk[:16].unsqueeze(0)  # (1, 16) as the prompt
        out = model.generate(prompt, num_new_tokens=num_new_tokens)
        text = tokenizer.decode(out[0].tolist())
        print("--- Generated sample ---")
        print(text)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="./model_tiny.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--num-new-tokens", type=int, default=64)
    parser.add_argument("--compare-gpt2", action="store_true", help="also load real GPT-2 for comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    batches = load_test_batches(
        TINY_CONFIG.context_length, args.num_batches, args.batch_size, device
    )

    # ---- Your model ----
    model = Transformer(TINY_CONFIG).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    avg_loss, ppl = evaluate_loss(model, batches)
    print(f"Your model  test loss = {avg_loss:.4f}   perplexity = {ppl:.2f}")
    generate_samples(
        model, tokenizer, device,
        num_samples=args.num_samples, num_new_tokens=args.num_new_tokens,
    )

    # ---- Real GPT-2 for comparison (optional) ----
    if args.compare_gpt2:
        try:
            gpt2 = Transformer.from_pretrained().to(device)
            gpt2_loss, gpt2_ppl = evaluate_loss(gpt2, batches)
            print(f"GPT-2       test loss = {gpt2_loss:.4f}   perplexity = {gpt2_ppl:.2f}")
        except Exception as e:  # e.g. offline, cannot download gpt2 weights
            print(f"Failed to load GPT-2 (check network / huggingface login): {e}")


if __name__ == "__main__":
    main()
