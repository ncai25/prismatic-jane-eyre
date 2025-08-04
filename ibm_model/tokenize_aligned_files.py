import spacy
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from spacy.lang.en import English
from spacy.lang.fr import French
from pathlib import Path
import spacy.util

def create_custom_tokenizer(nlp):
    """
    Create a tokenizer that preserves single hyphens but splits double dashes.
    Also splits connected punctuation like !-- into separate tokens.
    """
    
    # Add custom suffix patterns to split connected punctuation 
    suffixes = nlp.Defaults.suffixes + [r'''!-''', r'''\?-''', r'''\.\.\.-+''']
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search
    
    # Custom infix patterns - split double dash but not single hyphen
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\\-\\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            # Split punctuation followed by double dash
            r"(?<=[{a}])([!?.,;:])(--)".format(a=ALPHA),
            # Split on double dash in any context
            r"--",
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer
    
    return nlp.tokenizer

def tokenize_file(input_path, output_path, tokenizer):
    """
    Tokenize a file while maintaining sentence alignment.
    """
    print(f"Processing {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for i, line in enumerate(infile, 1):
            line = line.strip()
            
            doc = tokenizer(line)
            
            tokens = [token.text.lower() for token in doc]
            
            tokenized_line = ' '.join(tokens)
            
            outfile.write(tokenized_line + '\n')
            
            if i % 1000 == 0:
                print(f"  Processed {i} lines...")
    
    print(f"  Completed! Total lines: {i}")

def main():
    nlp_en = English()
    tokenizer_en = create_custom_tokenizer(nlp_en)
    
    nlp_fr = French()
    tokenizer_fr = nlp_fr
    
    base_dir = Path("jane-eyre/french")
    
    tokenize_file(
        base_dir / "fr_combined_aligned.f",
        base_dir / "fr_combined_aligned.f.tokenized",
        tokenizer_fr
    )
    
    tokenize_file(
        base_dir / "fr_combined_aligned.e",
        base_dir / "fr_combined_aligned.e.tokenized",
        tokenizer_en
    )
    
    print("\nTokenization complete!")
    print("Created files:")
    print("  - jane-eyre/french/fr_combined_aligned.f.tokenized")
    print("  - jane-eyre/french/fr_combined_aligned.e.tokenized")

if __name__ == "__main__":
    main()