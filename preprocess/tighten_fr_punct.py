import re
from pathlib import Path

def attach_punctuation(text):
    """Attach punctuation to adjacent words."""
    # Attach trailing punctuation to preceding words (including period before closing quote)
    text = re.sub(r'(\w)\s+([.,!?;:)])', r'\1\2', text)

    # Attach closing quotes and exclamation marks to preceding punctuation or words
    text = re.sub(r'([.?!…★",”)»–\w])\s+([»])', r'\1\2', text)
    text = re.sub(r'([.?!…★",)»”–\w])\s+([?])', r'\1\2', text)
    text = re.sub(r'([.?!…★",)»–\w])\s+([!])', r'\1\2', text)
    text = re.sub(r'([.?!…★",”\])»–\w])\s+([:;…*])', r'\1\2', text)

    # Attach opening punctuation to following words
    text = re.sub(r'([«(—])\s+(\w)', r'\1\2', text)
    # Also attach opening quotes to preceding dashes
    text = re.sub(r'(—)\s+([«’(“])', r'\1\2', text)
    # Attach em dash to preceding closing quote
    text = re.sub(r'(»)\s+(—)', r'\1\2', text)
    # Attach ellipsis and following words to em dash
    text = re.sub(r'(—)\s+(\.{2,3})\s+(\w)', r'\1\2\3', text)
    return text

def process_file(input_path, output_path):
    """Process a file to attach punctuation to adjacent words."""
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(attach_punctuation(line.strip()) + '\n')

if __name__ == "__main__":
    aligned_dir = Path("preprocess/aligned_outputs")
    tightened_dir = Path("preprocess/tightened_fr")
    tightened_dir.mkdir(exist_ok=True)
    
    process_file(
        aligned_dir / "fr_souvestre_aligned.txt",
        tightened_dir / "fr_souvestre_aligned.txt"
    )