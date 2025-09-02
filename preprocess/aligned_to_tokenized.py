import spacy
from spacy.lang.en import English
from spacy.lang.fr import French
from pathlib import Path

def strip_attached_punctuation(token_text):
    """
    Strip punctuation from the beginning and end of a token,
    but only if the token contains alphabetic characters.
    This preserves standalone punctuation tokens.
    """
    # If token is purely punctuation, keep it as is
    if not any(c.isalpha() for c in token_text):
        return token_text
    
    # If token contains alphabetic characters, strip punctuation from edges
    # Strip from the beginning
    start_idx = 0
    for i, char in enumerate(token_text):
        if char.isalpha() or char == '-':  # Keep hyphens within words
            start_idx = i
            break
    
    # Strip from the end
    end_idx = len(token_text)
    for i in range(len(token_text) - 1, -1, -1):
        if token_text[i].isalpha() or (i > 0 and token_text[i] == '-' and i < len(token_text) - 1):
            end_idx = i + 1
            break
    
    return token_text[start_idx:end_idx]

def tokenize_file(input_path, output_path, nlp):
    """
    Tokenize a file while maintaining sentence alignment.
    Strips attached punctuation from words but keeps standalone punctuation tokens.
    Converts to lowercase.
    """
    print(f"Processing {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile, 1):
            line = line.strip()
            
            # Use spaCy's default tokenizer
            doc = nlp(line)
            
            tokens = []
            for token in doc:
                # Strip attached punctuation from the token
                cleaned_token = strip_attached_punctuation(token.text)
                
                # If after stripping we still have content, keep it
                if cleaned_token:
                    tokens.append(cleaned_token.lower())
            
            tokenized_line = ' '.join(tokens)
            outfile.write(tokenized_line + '\n')
    
    print(f"  Completed! Total lines: {i}")

def main():
    # Just use the default tokenizers
    nlp_en = English()
    nlp_fr = French()
    
    aligned_text_dir = Path("preprocess/aligned_outputs")
    tokenized_dir = Path("preprocess/tokenized")
    tokenized_dir.mkdir(exist_ok=True)
    
    # Test examples to verify behavior
    print("\nTesting French tokenization:")
    test_sentences = [
        "« Que Bessie me reproche-t-elle ? demandai-je.",
        "J'en étais ravie.",
        "Mrs. Reed déjeunait tôt)",
    ]
    
    for sent in test_sentences:
        doc = nlp_fr(sent)
        tokens = []
        for token in doc:
            cleaned = strip_attached_punctuation(token.text)
            if cleaned:
                tokens.append(cleaned.lower())
        print(f"  Input: {sent}")
        print(f"  Output: {' '.join(tokens)}\n")
    
    tokenize_file(
        aligned_text_dir / "en_jean_aligned.txt",
        tokenized_dir / "en_jean_tokenized.txt",
        nlp_en
    )
    
    tokenize_file(
        aligned_text_dir / "fr_jean_aligned.txt",
        tokenized_dir / "fr_jean_tokenized.txt",
        nlp_fr
    )
        
if __name__ == "__main__":
    main()