import requests
from new_hindi_tokenizer import SimpleHindiTokenizer


VOCAB_SIZE = 5000
merges = 4743

def fetch_hindi_instruct_data():
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": "maharnab/hindi_instruct",
        "config": "default",
        "split": "train",
        "offset": 0,
        "length": 100  # Fetch 100 rows to ensure we have enough data
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def process_data(data):
    rows = data['rows']
    num_rows = len(rows)
    sample_size = int(num_rows * 1)  # 30% of the data
    sample_data = rows[:sample_size]

    # Display the first 5 rows
    for i, row in enumerate(sample_data[:5]):
        print(f"Row {i+1}: {row['row']['output']}")

    # Concatenate all text from the sample data
    concatenated_text = " ".join(row['row']['output'] for row in sample_data)
    
    # Calculate statistics
    total_length = len(concatenated_text)
    num_characters = len(concatenated_text)  # This should match total_length
    num_words = len(concatenated_text.split())

    print(f"Total length of text: {total_length}")
    print(f"Number of characters: {num_characters}")
    print(f"Number of words: {num_words}")

    # Tokenize the text using the Hindi tokenizer
    tokenizer = SimpleHindiTokenizer()
#    original_tokens = tokenizer.tokenize_hindi(concatenated_text)
#    print(f"Original token count (before BPE): {len(original_tokens)}")

    # Calculate the initial length of the original tokens
#   initial_length = sum(len(token) for token in original_tokens)
#  print(f"Initial length of original tokens (before BPE): {initial_length}")

    # Print the first 100 original tokens
#   print("First 100 original tokens:", original_tokens[:100])

    # Learn BPE vocabulary and tokenize using BPE
    tokenizer.learn_bpe_vocab(concatenated_text)
    bpe_tokens = tokenizer.tokenize_bpe(concatenated_text)
    print(f"BPE token count: {len(bpe_tokens)}")

    # Calculate and print the compression ratio
 #   compression_ratio = len(original_tokens) / len(bpe_tokens)
  #  print(f"Compression ratio: {compression_ratio:.2f}")

    # Save the learned vocabulary
    tokenizer.save_bpe_vocab("hindi_bpe_vocab.model")
    tokenizer.load_bpe_vocab("hindi_bpe_vocab.model")


def main():
    data = fetch_hindi_instruct_data()
    process_data(data)

if __name__ == "__main__":
    main()