"""Byte-pair encoding over UTF-8 bytes with lossless fallback.

This tokenizer learns frequent byte sequences from the training corpus,
keeps single bytes for exact round-tripping, and greedily encodes unseen
text using the saved vocabulary.
"""
import json
import os
from collections import Counter

SAVE_PATH = os.path.join(os.path.dirname(__file__), "tokenizer_bpe.json")
TRAIN_CORPUS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "train_corpus.txt"))
WHITESPACE_BYTES = {9, 10, 11, 12, 13, 32}
DEFAULT_VOCAB_SIZE = 1920
TRAIN_SAMPLE_BYTES = 256_000

class BytePairTokenizer:
    def __init__(self, merges=None):
        self.tokens = [bytes([i]) for i in range(256)]
        self.merges = merges or []
        for a, b in self.merges:
            self.tokens.append(a + b)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)
        self.max_token_len = max(len(tok) for tok in self.tokens)

    def encode(self, text):
        data = text.encode("utf-8")
        ids = []
        i = 0
        while i < len(data):
            end = min(len(data), i + self.max_token_len)
            while end > i:
                piece = data[i:end]
                token_id = self.token_to_id.get(piece)
                if token_id is not None:
                    ids.append(token_id)
                    i = end
                    break
                end -= 1
            else:
                raise ValueError("Unable to tokenize byte sequence")
        return ids

    def decode(self, ids):
        data = bytearray()
        for idx in ids:
            data.extend(self.tokens[idx])
        return data.decode("utf-8", errors="strict")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "merges": [[list(a), list(b)] for a, b in self.merges],
            }, f, separators=(",", ":"))

    @staticmethod
    def _word_counts(data):
        word_counts = Counter()
        word = []
        for b in data:
            if b in WHITESPACE_BYTES:
                if word:
                    word_counts[tuple(word)] += 1
                    word = []
                word_counts[(bytes([b]),)] += 1
            else:
                word.append(bytes([b]))
        if word:
            word_counts[tuple(word)] += 1
        return word_counts

    @classmethod
    def train(cls, text, vocab_size=DEFAULT_VOCAB_SIZE):
        data = text.encode("utf-8")
        if len(data) > TRAIN_SAMPLE_BYTES:
            data = data[:TRAIN_SAMPLE_BYTES]
        word_counts = cls._word_counts(data)
        merges = []
        for _ in range(vocab_size - 256):
            pair_counts = Counter()
            for word, freq in list(word_counts.items()):
                if len(word) < 2:
                    continue
                for a, b in zip(word, word[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            (a, b), count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            merges.append((a, b))
            next_word_counts = Counter()
            for word, freq in list(word_counts.items()):
                if len(word) < 2:
                    next_word_counts[word] += freq
                    continue
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                next_word_counts[tuple(new_word)] += freq
            word_counts = next_word_counts
        tokenizer = cls(merges=merges)
        tokenizer.vocab_size = len(tokenizer.tokens)
        tokenizer.max_token_len = max(len(tok) for tok in tokenizer.tokens)
        return tokenizer

def load(path=None):
    if path is None:
        path = SAVE_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = [(bytes(a), bytes(b)) for a, b in data["merges"]]
        return BytePairTokenizer(merges=merges)
    if os.path.exists(TRAIN_CORPUS_PATH):
        text = open(TRAIN_CORPUS_PATH, encoding="utf-8").read()
        data = text.encode("utf-8")
        if len(data) > TRAIN_SAMPLE_BYTES:
            print(f"Training tokenizer from first {TRAIN_SAMPLE_BYTES:,} bytes", flush=True)
        else:
            print(f"Training tokenizer from full corpus ({len(data):,} bytes)", flush=True)
        tokenizer = BytePairTokenizer.train(text, vocab_size=DEFAULT_VOCAB_SIZE)
        print(f"Tokenizer built with {tokenizer.vocab_size} tokens", flush=True)
        tokenizer.save(path)
        return tokenizer
    raise FileNotFoundError(
        f"Tokenizer file not found at {path} and training corpus is missing.")