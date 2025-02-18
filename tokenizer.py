from typing import List, Tuple, Dict, Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        
        # Extend vocabulary with special tokens
        for token in self.special_tokens:
            if token.encode("utf-8") not in self.vocab.values():
                self.vocab[len(self.vocab)] = token.encode("utf-8")
        
        # Create a reverse vocabulary lookup for decoding
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        """
        Constructs and returns a Tokenizer from serialized vocabulary and merge files.
        """
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab = {int(k): bytes.fromhex(v.strip()) for k, v in (line.split() for line in vf)}

        with open(merges_filepath, "r", encoding="utf-8") as mf:
            merges = [tuple(line.strip().split()) for line in mf]
            merges = [(a.encode("utf-8"), b.encode("utf-8")) for a, b in merges]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        """
        Encodes an input text into a sequence of token IDs.
        """
        tokens = [char.encode("utf-8") for char in text]
        
        # Merge token pairs based on the BPE merges
        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            merge_idx = {pair: self.merges.get(pair, float("inf")) for pair in pairs}
            best_pair = min(merge_idx, key=merge_idx.get)
            
            if best_pair not in self.merges:
                break  # No more merges available

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(b"".join(best_pair))
                    i += 2  # Skip the next token as it's merged
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return [self.reverse_vocab[token] for token in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a sequence of token IDs into text.
        """
        return "".join(self.vocab[i].decode("utf-8") for i in ids)
