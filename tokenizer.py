import numpy as np
import math
import regex as re

'''
- Let's handle one chunk for now
'''

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab = vocab
        
        self.special_tokens = [] if not special_tokens else special_tokens
        if self.special_tokens:
            specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = "(" + "|".join(re.escape(k) for k in specials_sorted) + ")"
        self.TOK = re.compile(PAT)
        self.rev_vocab = {}
        for k, v in self.vocab.items():
            self.rev_vocab[v] = k
        
        self.merges = {pair: i for i, pair in enumerate(merges)}
        

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        return 
    

    def get_pairs(self, bytes_tup):
        pairs = set()
        for i in range(len(bytes_tup)-1):
            pairs.add((bytes_tup[i], bytes_tup[i+1], i))
        
        return pairs

    
    def get_encoding(self, match):
        encoded_tuple = []
        bytes_tup = tuple([bytes([b]) for b in match.group(0).encode("utf-8")])

        while len(bytes_tup) > 1:
            pairs = self.get_pairs(bytes_tup)
            pair_to_merge = min(pairs, key = lambda x: self.merges.get(x[:-1], float('inf')))
            if pair_to_merge[:-1] not in self.merges:
                break
            idx = pair_to_merge[-1]

            bytes_tup = bytes_tup[:idx] + (pair_to_merge[0]+pair_to_merge[1],) + bytes_tup[idx+2:]
        
        for obj in bytes_tup:
            encoded_tuple.append(self.rev_vocab[obj])

        return encoded_tuple


    def encode(self, text):
        encoded_arr = []
        # vocab -> dict[int, bytes]
        # merges -> list[tuple[bytes, bytes]]

        # 1. add special tokens to the vocab
        vocab_values = self.vocab.values()
        vocab_len = len(self.vocab)-1
        for s in self.special_tokens:
            if s not in vocab_values:
                self.vocab[vocab_len] = s.encode("utf-8")
                self.rev_vocab[s] = vocab_len
                vocab_len += 1

        # 2. merge and encode
        if self.special_tokens:
            chunks = re.split(self.special_pattern, text)
        else:
            chunks = [text]
        for c in chunks:
            if not c:
                continue
            if c in self.special_tokens:
                encoded_arr += [self.rev_vocab[c]]
                continue
            for m in self.TOK.finditer(c):
                # breakpoint()
                if m.group(0).encode("utf-8") in self.rev_vocab:
                    encoded_arr.append(self.rev_vocab[m.group(0).encode("utf-8")])
                else:
                    encoded_arr += self.get_encoding(m)
        
        return encoded_arr


    def encode_iterable(self, iterable):
        for t in iterable: 
            yield from self.encode(t)
    

    def decode(self, ids):
        final_text = b''

        for id in ids:
            final_text += self.vocab[id]

        return final_text.decode("utf-8", errors="replace")


'''
vocab -> dict[int, bytes]
merges -> list[tuple[bytes, bytes]]
'''