import json
import regex
from typing import Iterable, Iterator
import time

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        
        # Add special tokens to vocab first
        if special_tokens:
            idx = len(self.vocab)
            for special_token in special_tokens:
                token_bytes = special_token.encode("utf-8")
                if token_bytes not in vocab.values():
                    vocab[idx] = token_bytes
                    idx += 1
        
        # Build reverse lookup AFTER adding special tokens
        self.byte2id = {v: k for k, v in self.vocab.items()}
        self.special_tokens = special_tokens or []



    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        
        with open(vocab_filepath, "r") as vf:
            vocab = {int(k): bytes.fromhex(v) for k, v in json.load(vf).items()}

        with open(merges_filepath, "r") as mf:
            raw_merges = json.load(mf)
            merges = [tuple(bytes.fromhex(a) for a in pair) for pair in raw_merges]

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        encoded_ids = []

        if self.special_tokens:
            # split **and keep** the delimiters
            specials = sorted(self.special_tokens, key=len, reverse=True)
            split_re = regex.compile("(" + "|".join(regex.escape(t) for t in specials) + ")")
            parts = split_re.split(text)
        else:
            parts = [text]

        merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        for part in parts:
            # Skip empty parts
            if not part:
                continue
                
            # 2a. special token => one ID, done
            if self.special_tokens and part in self.special_tokens:
                tok_id = self.byte2id[part.encode("utf-8")]
                encoded_ids.append(tok_id)
                continue

            # 2b. regular text => BPE tokenize
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pre_tokens = [match.group(0) for match in regex.finditer(PAT, part)]

            def apply_merges(bytes_tuple):
                while True:
                    # find first ranked pair present
                    candidates = [(merge_ranks[(bytes_tuple[i], bytes_tuple[i+1])], i)
                                for i in range(len(bytes_tuple)-1)
                                if (bytes_tuple[i], bytes_tuple[i+1]) in merge_ranks]
                    if not candidates:
                        break
                    _, i = min(candidates)  # lowest rank wins
                    bytes_tuple = (bytes_tuple[:i] +
                                (bytes_tuple[i] + bytes_tuple[i+1],) +
                                bytes_tuple[i+2:])
                return bytes_tuple
            

            for token in pre_tokens:
                byte_seq = tuple(bytes([b]) for b in token.encode('utf-8'))
                bytes_tuple = apply_merges(byte_seq)

                for token_bytes in bytes_tuple:
                    encoded_ids.append(self.byte2id[token_bytes])

        return encoded_ids


    def encode_iterable(self, iterable: Iterable[str]):
        for chunk in iterable:              # `chunk` already includes its '\n'
            for tok in self.encode(chunk):  # reuse the regular encoder
                yield tok


    def decode(self, ids: list[int]) -> str:
        # gather *all* bytes, then decode once
        joined: bytes = b"".join(self.vocab[_id] for _id in ids)
        return joined.decode("utf‑8", errors="replace")


def count_lines(filepath):
    """Count total lines in file for progress tracking"""
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)


import multiprocessing
import numpy as np
if __name__ == "__main__":
    # ===== CONFIGURATION =====
    DATASET = "val"  # Change this! "train" or "val" 
    PERCENTAGE = 0.1  # Change this! 0.01 = 1%, 0.1 = 10%, 1.0 = 100%
    
    # Examples:
    # DATASET="train", PERCENTAGE=0.01  → ../data/TinyStoriesV2-GPT4-train-tok-1pct.npy
    # DATASET="val", PERCENTAGE=0.1     → ../data/TinyStoriesV2-GPT4-valid-tok-10pct.npy
    # DATASET="train", PERCENTAGE=1.0   → ../data/TinyStoriesV2-GPT4-train-tok.npy
    
    tokenizer = Tokenizer.from_files("vocab.json", "merges.json", special_tokens=["<|endoftext|>"])

    # Set paths based on dataset choice
    if DATASET == "train":
        input_path = "TinyStoriesV2-GPT4-train.txt"
        base_output_path = "../data/TinyStoriesV2-GPT4-train-tok"
    elif DATASET == "val":
        input_path = "data/TinyStoriesV2-GPT4-valid.txt"
        base_output_path = "data/TinyStoriesV2-GPT4-valid-tok"
    else:
        raise ValueError(f"Invalid dataset '{DATASET}'. Must be 'train' or 'val'")
    
    output_path = f"{base_output_path}.npy"

    # Count total lines for progress tracking
    print("Counting total lines...")
    total_lines = count_lines(input_path)
    lines_to_process = int(total_lines * PERCENTAGE)
    
    print(f"Total lines in file: {total_lines:,}")
    print(f"Processing {PERCENTAGE*100:.1f}% = {lines_to_process:,} lines")
    print(f"Dataset: {DATASET}")
    
    if PERCENTAGE < 1.0:
        # Update output filename to include percentage
        output_path = f"{base_output_path}-{PERCENTAGE*100:.0f}pct.npy"
    
    print(f"Output file: {output_path}")

    ids = []
    lines_processed = 0
    start_time = time.time()
    
    # More frequent logging, especially at the beginning
    def should_log(lines_processed, lines_to_process):
        if lines_processed <= 1000:
            return lines_processed % 100 == 0  # Every 100 lines for first 1000
        elif lines_processed <= 10000:
            return lines_processed % 1000 == 0  # Every 1000 lines for first 10k
        else:
            # After 10k lines, log every 1% or 10k lines, whichever is smaller
            interval = min(10000, max(1000, lines_to_process // 100))
            return lines_processed % interval == 0

    print(f"Starting tokenization... (will log every 100 lines initially)")

    with open(input_path, "r") as f:
        for line in f:
            if not line:
                continue
            
            # Stop if we've reached our target percentage
            if lines_processed >= lines_to_process:
                break
        
            ids.extend(tokenizer.encode(line))
            lines_processed += 1
            
            # Log progress with adaptive frequency
            if should_log(lines_processed, lines_to_process):
                elapsed = time.time() - start_time
                progress_pct = (lines_processed / lines_to_process) * 100
                lines_per_sec = lines_processed / elapsed if elapsed > 0 else 0
                tokens_so_far = len(ids)
                
                # Estimate time remaining
                if lines_per_sec > 0:
                    remaining_lines = lines_to_process - lines_processed
                    eta_seconds = remaining_lines / lines_per_sec
                    if eta_seconds < 60:
                        eta_str = f", ETA: {eta_seconds:.0f}s"
                    elif eta_seconds < 3600:
                        eta_str = f", ETA: {eta_seconds/60:.1f}m"
                    else:
                        eta_str = f", ETA: {eta_seconds/3600:.1f}h"
                else:
                    eta_str = ""
                
                avg_tokens_per_line = tokens_so_far / lines_processed if lines_processed > 0 else 0
                
                print(f"Progress: {lines_processed:,}/{lines_to_process:,} lines ({progress_pct:.1f}%) | "
                      f"{tokens_so_far:,} tokens ({avg_tokens_per_line:.1f} tok/line) | "
                      f"{lines_per_sec:.0f} lines/sec{eta_str}")

    # Final summary
    elapsed = time.time() - start_time
    final_tokens = len(ids)
    print(f"\nTokenization complete for {DATASET} dataset!")
    print(f"Lines processed: {lines_processed:,} ({PERCENTAGE*100:.1f}% of total)")
    print(f"Total tokens: {final_tokens:,}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average: {lines_processed/elapsed:.0f} lines/sec, {final_tokens/elapsed:.0f} tokens/sec")

    ids = np.array(ids)
    np.save(output_path, ids)
    print(f"Saved to {output_path}")