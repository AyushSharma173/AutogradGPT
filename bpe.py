from typing import List, Tuple
from utils import find_chunk_boundaries

from collections import defaultdict
import multiprocessing
import regex

from tqdm import tqdm

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]):

    with open(input_path, "rb") as f:
        
        sentinels = [tok.encode("utf-8") for tok in special_tokens]
        if b"<|endoftext|>" not in sentinels:        # always include canonical EOT
            sentinels.append(b"<|endoftext|>")
        boundaries = find_chunk_boundaries(f, 4000, *sentinels)
        print(f"Found {len(boundaries)} chunk boundaries")
                    
        
        vocab = {i: bytes([i]) for i in range(256)}
        idx = 256
        for special_token in special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in vocab.values():
                vocab[idx] = token_bytes
                idx += 1
            else:
                # Optional: you can log or store the existing ID if needed
                pass

        corpus = defaultdict(int)

        chunk_spans = chunk_spans = list(zip(boundaries[:-1], boundaries[1:]))

        args = [(start, end, input_path, special_tokens) for start, end in chunk_spans]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            print(f"Starting multiprocessing pool on {len(chunk_spans)} chunks...")
            results = pool.map(process_chunk, args)

        
        global_corpus = defaultdict(int)
        for local_corpus in results:
            for token_bytes, count in local_corpus.items():
                global_corpus[token_bytes] += count
                

        
        def merge_sequence(seq: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
            A, B = pair
            merged = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == A and seq[i + 1] == B:
                    merged.append(A + B)
                    i += 2
                else:
                    merged.append(seq[i])
                    i += 1
            return tuple(merged)
        
        merges = []
        
        pbar = tqdm(total=vocab_size - len(vocab), desc="Merging BPE pairs")
        while len(vocab) < vocab_size:

            pair_counts = defaultdict(int)
            
            for seq, freq in global_corpus.items():
                # skip single-byte sequences (saves a tiny bit)
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i+1])] += freq

            if not pair_counts:
                break

            most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

            merged_token = most_frequent_pair[0] + most_frequent_pair[1]

            vocab[idx] = merged_token

            merges.append(most_frequent_pair)

            idx += 1

            # Log top pair and current vocab size occasionally
            if idx % 500 == 0 or idx < 300:
                print(f"[merge {idx}] top pair: {most_frequent_pair} → {merged_token} (vocab size: {len(vocab)})")

            # Update progress bar
            pbar.update(1)


            new_corpus = defaultdict(int)
            for seq, freq in global_corpus.items():
                new_seq = merge_sequence(seq, most_frequent_pair)
                new_corpus[new_seq] += freq

            global_corpus = new_corpus

            # Update progress bar
        pbar.close()

    return vocab, merges


        


def process_chunk(args):
    start, end, input_path, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
    # remove special tokens
    pattern = regex.compile("|".join(regex.escape(tok) for tok in special_tokens))
    # chunk = pattern.sub("", chunk)
    parts = regex.split(pattern, chunk)
    total = len(parts)
    local_corpus = defaultdict(int)
    for part in parts:
        if not part:
            continue
        # run regex pretokenization
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        matches = regex.finditer(PAT, part)

        
        for match in matches:
            pretoken = match.group(0)
            # pretoken_bytes = tuple(pretoken.encode("utf-8"))
            pretoken_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
            local_corpus[pretoken_bytes] += 1
            total += 1

    print(f"Processed chunk from {start} to {end}: {total} pre-tokens")
    return local_corpus




def test_debug():
    vocab = {i: bytes([i]) for i in range(256)}
    print(vocab)
    pass


import json

if __name__ == "__main__":
    # ✅ Path to raw training text
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"

    # ✅ Max vocab size including special tokens and byte vocab
    vocab_size = 10_000

    # ✅ Must include this token for TinyStories
    special_tokens = ["<|endoftext|>"]

    # ✅ Train BPE tokenizer
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    print(f"vocab size: {len(vocab)} • merges: {len(merges)}")

    # ✅ Save vocab (convert bytes -> hex string so it's JSON serializable)
    vocab_json = {str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()}
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, indent=2)

    # ✅ Save merges (convert list[tuple[bytes, bytes]] -> list[tuple[str, str]])
    merges_json = [(t1.hex(), t2.hex()) for (t1, t2) in merges]
    with open("merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_json, f, indent=2)