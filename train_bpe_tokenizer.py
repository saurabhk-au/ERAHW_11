import re
from collections import defaultdict, Counter

# Read the Hindi text from the file
with open("hindi_text.txt", "r", encoding="utf-8") as f:
    hindi_text = f.read()

# Tokenize the text into characters
tokens = list(hindi_text)

# Initialize vocabulary with unique characters
vocab = set(tokens)

# Function to get the most frequent pair of tokens
def get_most_frequent_pair(tokens):
    pairs = defaultdict(int)
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pairs[pair] += 1
    return max(pairs, key=pairs.get)

# Function to merge a pair in the token list
def merge_pair(tokens, pair):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

# Set desired vocabulary size and target token count
desired_vocab_size = 5000
target_token_count = 1163

# Perform BPE until the vocabulary size is reached or token count is within target
pair_vocab = []
while len(vocab) < desired_vocab_size and len(tokens) > target_token_count:
    pair = get_most_frequent_pair(tokens)
    tokens = merge_pair(tokens, pair)
    vocab.update(tokens)
    pair_vocab.append(pair)

# Save the pair vocabulary with token numbers
with open("pair_vocab.txt", "w", encoding="utf-8") as f:
    for i, pair in enumerate(pair_vocab):
        f.write(f"{pair[0]}{pair[1]} {i}\n")

# Calculate and print final statistics
final_vocab_size = len(vocab)
encoded_token_count = len(tokens)
compression_ratio = len(hindi_text) / encoded_token_count
sentence_count = hindi_text.count('.') + hindi_text.count('।')
word_count = len(hindi_text.split())

# Decode the encoded text
decoded = ''.join(tokens)

# Update README.md
readme_content = f"""
# Hindi BPE Tokenizer

This project trains a Byte Pair Encoding (BPE) tokenizer on a sample Hindi text.

## Output

### Final Vocabulary Size
- **Vocabulary Size**: {final_vocab_size}

### Pair Vocabulary with Token Numbers
"""

# Print the final vocabulary and pair vocabulary
print("Final Vocabulary Size:", len(vocab))
print("Final Pair Vocabulary with Token Numbers:", [(pair, i) for i, pair in enumerate(pair_vocab)])

# Encode the text using the final vocabulary
encoded = [tokens.index(token) for token in tokens]
print("Encoded Tokens Count:", len(encoded))

# Calculate and print compression ratio
compression_ratio = len(hindi_text) / len(encoded)
print(f"Compression Ratio: {compression_ratio:.2f}")

# Calculate and print sentence and word counts
sentence_count = hindi_text.count('.') + hindi_text.count('।')
word_count = len(hindi_text.split())

print(f"Sentence Count: {sentence_count}")
print(f"Word Count: {word_count}")

# Decode the encoded text
decoded = ''.join(tokens)
print("Decoded Text:", decoded)

# Calculate and print original characters, words, spaces
original_characters = len(hindi_text)
original_words = len(hindi_text.split())
original_spaces = hindi_text.count(' ')

print(f"Original Characters: {original_characters}")
print(f"Original Words: {original_words}")
print(f"Original Spaces: {original_spaces}")

# Create character mappings
chars = sorted(set(hindi_text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Define encode and decode functions
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Test the custom encode and decode functions
custom_encoded = encode(hindi_text)
print("Custom Encoded:", custom_encoded)

custom_decoded = decode(custom_encoded)
print("Custom Decoded Text:", custom_decoded)

# Example: Find all Hindi words
hindi_words = re.findall(r'\b[ऀ-ॿ]+\b', hindi_text)
print("Hindi Words:", hindi_words)
