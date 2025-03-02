import regex as re
from collections import defaultdict
from tqdm import tqdm

VOCAB_SIZE = 5000  # Total desired vocabulary size
BASE_VOCAB_SIZE = 256  # Initial vocabulary size (first 256 Unicode bytes)
Merges = 4743  # Number of merges to perform

class SimpleHindiTokenizer:
    def __init__(self):
        self.pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{N}+| ?(?:[\u0904-\u0939\u093d-\u093d\u0950-\u0950\u0958-\u0961\u0970-\u097f\ua8f2-\ua8fe\U00011b00-\U00011b09\u1cd3-\u1cd3\u1ce9-\u1cec\u1cee-\u1cf3\u1cf5-\u1cf6\u1cfa-\u1cfa][\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\ua8e0-\ua8f1\ua8ff-\ua8ff\u1cd0-\u1cd2\u1cd4-\u1ce8\u1ced-\u1ced\u1cf4-\u1cf4\u1cf7-\u1cf9]*)+| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""";
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(BASE_VOCAB_SIZE)}
        self.special_tokens = {'<|endoftext|>': VOCAB_SIZE}

    def tokenize_hindi(self, text):
        pattern = re.compile(self.pattern)
        return pattern.findall(text)

    def get_stats(self, ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def learn_bpe_vocab(self, text, num_merges=2882):
        # Tokenize the text using the Hindi tokenizer
        tokenized_text = self.tokenize_hindi(text)
        # Convert tokens to a list of integer lists
        token_ids = [list(map(int, token.encode("utf-8"))) for token in tokenized_text]

        # Calculate the initial input length for compression ratio
        initial_length = sum(len(chunk) for chunk in token_ids)

        # Perform BPE merges
        for merge_step in tqdm(range(num_merges), desc="Merging pairs", unit="merge"):
            # Collect statistics on pairs of tokens
            pair_stats = defaultdict(int)
            for chunk in token_ids:
                pair_stats = self.get_stats(chunk, pair_stats)

            # Find the most frequent pair to merge
            if not pair_stats:
                print("No more pairs to merge.")
                break
            most_frequent_pair = max(pair_stats, key=pair_stats.get)

            # Assign a new index for the merged pair
            new_index = BASE_VOCAB_SIZE + merge_step
            token_ids = [self.merge(chunk, most_frequent_pair, new_index) for chunk in token_ids]

            # Update the merges and vocabulary
            self.merges[most_frequent_pair] = new_index
            self.vocab[new_index] = self.vocab[most_frequent_pair[0]] + self.vocab[most_frequent_pair[1]]

        # Calculate the output length for compression ratio
        final_length = sum(len(chunk) for chunk in token_ids)

        # Print the compression ratio
        print(f"Initial length: {initial_length}, Final length: {final_length}, Compression ratio: {initial_length / final_length:.2f}X")

    def encode(self, text):
        text_chunks = self.tokenize_hindi(text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids

    def tokenize_bpe(self, text):
        # Tokenize the text using the Hindi tokenizer
        tokenized_text = self.tokenize_hindi(text)
        # Convert tokens to a list of integer lists
        token_ids = [list(map(int, token.encode("utf-8"))) for token in tokenized_text]

        # Encode each chunk using the BPE vocabulary
        bpe_encoded_ids = []
        for chunk in token_ids:
            encoded_chunk = self._encode_chunk(chunk)
            bpe_encoded_ids.extend(encoded_chunk)

        return bpe_encoded_ids

    def _encode_chunk(self, chunk):
        # Encode a single chunk using the BPE vocabulary
        while len(chunk) >= 2:
            # Collect statistics on pairs of tokens
            stats = self.get_stats(chunk)
            # Find the pair with the lowest merge index
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # Check if the pair can be merged
            if pair not in self.merges:
                break
            # Merge the pair
            idx = self.merges[pair]
            chunk = self.merge(chunk, pair, idx)
        return chunk

    def save_bpe_vocab(self, filepath='hindi_bpe_vocab.model'):
        with open(filepath, 'w', encoding='utf-8') as f:
            for idx, token in sorted(self.vocab.items()):
                try:
                    # Use repr to ensure byte strings are written correctly
                    token_str = repr(token)
                except UnicodeEncodeError:
                    token_str = str(token)
                f.write(f"{idx}: {token_str}\n")
        print(f"Vocabulary saved to {filepath}")

    def load_bpe_vocab(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.merges = {}
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
            for line in f:
                line = line.strip()
                if ': ' in line:
                    idx, token = line.split(': ', 1)
                    idx = int(idx)
                    try:
                        token_bytes = token.encode('utf-8')
                    except UnicodeEncodeError:
                        token_bytes = eval(token)  # Use eval to handle byte strings
                    self.vocab[idx] = token_bytes
                    if len(token_bytes) > 1:
                        self.merges[(token_bytes[0], token_bytes[1])] = idx
                else:
                    print(f"Skipping malformed line: {line}")

class SimpleTokenizer:
    def tokenize(self, text):
        # Simple whitespace tokenizer
        return text.split()

# Example usage
#if __name__ == "__main__":
 #   tokenizer = SimpleHindiTokenizer()
  #  sample_text = "यह एक उदाहरण पाठ है।"  # Example Hindi text
   # tokenizer.learn_bpe_vocab(sample_text)
