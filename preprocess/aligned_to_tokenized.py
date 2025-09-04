import spacy
from spacy.lang.en import English
from spacy.lang.fr import French
from pathlib import Path

def process_token(token_text):
    """Keep only tokens with letters, preserve contractions but remove attached punctuation."""
    # Only keep tokens that contain at least one letter
    if not any(c.isalpha() for c in token_text):
        return ""
    
    import re
    # For contractions, keep the apostrophe
    if "'" in token_text and re.search(r"\w'\w", token_text):
        # Remove leading/trailing punctuation except apostrophes in contractions
        cleaned = re.sub(r'^[^\w\']+|[^\w\']+$', '', token_text)
    else:
        # For regular words, remove ALL leading and trailing punctuation
        cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', token_text)
    
    if cleaned and any(c.isalpha() for c in cleaned):
        return cleaned.lower()
    return ""

def preprocess_french(text):
    """Join hyphenated forms in French text."""
    import re
    # Fix apostrophe contractions: l' → l', d' → d', qu' → qu'
    text = re.sub(r"(\w)'\s+", r"\1'", text)  
    # Fix hyphenated verb forms: -je, -il, -elle, -ce, -t-
    text = re.sub(r'\s+-\s*', r'-', text)  # Remove spaces around hyphens
    text = re.sub(r'\s*-\s+', r'-', text)  # Handle any remaining patterns
    return text

def preprocess_english(text):
    """Join contractions and hyphenated words in English text."""
    import re
    # Remove em dashes (--) entirely
    text = re.sub(r'--+', ' ', text)
    # Fix contractions: do n't → don't, wo n't → won't, etc.
    text = re.sub(r"(\w)\s+n't", r"\1n't", text)
    # Fix other contractions: I 'm → I'm, he 's → he's, etc.
    text = re.sub(r"(\w)\s+'([a-z]+)", r"\1'\2", text)
    # Fix hyphenated words: drawing - room → drawing-room
    text = re.sub(r'(\w)\s+-\s+(\w)', r'\1-\2', text)
    return text

def tokenize_file(input_path, output_path, nlp, is_french=False, is_english=False):
    """
    Tokenize a file while maintaining sentence alignment.
    Removes pure punctuation, preserves contractions, converts to lowercase.
    """
    print(f"Processing {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile, 1):
            line = line.strip()
            
            if is_french:
                line = preprocess_french(line)
            elif is_english:
                line = preprocess_english(line)
            
            doc = nlp(line)
            
            tokens = []
            for token in doc:
                processed = process_token(token.text)
                if processed:
                    tokens.append(processed)
            
            tokenized_line = ' '.join(tokens)
            outfile.write(tokenized_line + '\n')
    
    print(f"  Completed! Total lines: {i}")

def main():
    nlp_en = English()
    nlp_fr = French()
    
    from spacy.tokenizer import Tokenizer
    nlp_en.tokenizer = Tokenizer(nlp_en.vocab)
    nlp_fr.tokenizer = Tokenizer(nlp_fr.vocab)
    
    aligned_text_dir = Path("preprocess/aligned_outputs")
    tokenized_dir = Path("preprocess/tokenized")
    tokenized_dir.mkdir(exist_ok=True)

    tokenize_file(
        aligned_text_dir / "en_souvestre_aligned.txt",
        tokenized_dir / "en_souvestre_tokenized.txt",
        nlp_en,
        is_english=True
    )
    
    tokenize_file(
        aligned_text_dir / "fr_souvestre_aligned.txt",
        tokenized_dir / "fr_souvestre_tokenized.txt",
        nlp_fr,
        is_french=True
    )
        
if __name__ == "__main__":
    main()