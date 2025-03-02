from new_hindi_tokenizer import SimpleHindiTokenizer

def main():
    # Load the text from a file
    with open('Hindi_Text.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # Initialize the tokenizer
    tokenizer = SimpleHindiTokenizer()

    # Tokenize the text using the Hindi tokenizer to get original tokens
    original_tokens = tokenizer.tokenize_hindi(text)
    
    # Convert original tokens to token IDs (UTF-8 byte representation)
    original_token_ids = [list(token.encode('utf-8')) for token in original_tokens]
    
    # Calculate the initial length based on token IDs
    initial_token_id_length = sum(len(token_ids) for token_ids in original_token_ids)

    print(f"Initial token ID length: {initial_token_id_length}")
    print("Original tokens:")
    print(original_tokens)  # Print all original tokens

    print("Original token IDs:")
    for token, token_ids in zip(original_tokens, original_token_ids):
        print(f"Token: {token}, IDs: {token_ids}")

    # Load the trained BPE model
    tokenizer.load_bpe_vocab('hindi_bpe_vocab.model')

    # Tokenize the text using the BPE model to get BPE-transformed tokens
    bpe_tokens = tokenizer.tokenize_bpe(text)
    final_token_length = len(bpe_tokens)

    print(f"Final BPE token count: {final_token_length}")
    print("BPE-transformed tokens:")
    print(bpe_tokens)  # Print all BPE-transformed tokens

    # Calculate and print the compression ratio
    if final_token_length > 0:
        compression_ratio = initial_token_id_length / final_token_length
        print(f"Compression ratio: {compression_ratio:.2f}")
    else:
        print("Error: Final BPE token count is zero.")

if __name__ == "__main__":
    main() 