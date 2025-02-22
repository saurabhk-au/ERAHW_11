# Hindi BPE Tokenizer

This project trains a Byte Pair Encoding (BPE) tokenizer on a sample Hindi text using the `sentencepiece` library.

## Output

### Final Vocabulary Size
- **Vocabulary Size**: [Final Vocabulary Size]

### Pair Vocabulary with Token Numbers

### Encoded Tokens

### Original Characters, Words, Spaces
- **Original Characters**: [Number]
- **Original Words**: [Number]
- **Original Spaces**: [Number]

### Encoded Tokens Count
- **Encoded Tokens Count**: [Encoded Tokens Count]

### Compression Ratio
- **Compression Ratio**: [Compression Ratio]

### Sentence and Word Counts
- **Sentence Count**: [Sentence Count]
- **Word Count**: [Word Count]

### Decoded Text

## Setup

1. Clone the repository.
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python train_bpe_tokenizer.py
   ```

## GitHub Actions

This project uses GitHub Actions to automatically run the script on every push or pull request to the `main` branch.

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)