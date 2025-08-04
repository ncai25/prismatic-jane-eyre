import re
import sys

def extract_aligned_sentences(name):
    """Extract aligned sentences from vecalign output into separate files"""
    
    src_sentences = []
    tgt_sentences = []
    
    # Construct input filename
    input_file = f'outputs/{name}_output.txt'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for alignment lines like "[0]:[0]:0.514871"
        if re.match(r'^\[\d+.*\]:\[\d+.*\]:', line):
            # Next lines should contain SRC and TGT
            i += 1
            
            # Get source sentence
            if i < len(lines) and 'SRC:' in lines[i]:
                src_line = lines[i].strip()
                src_sentence = src_line.split('SRC:', 1)[1].strip()
                src_sentences.append(src_sentence)
                i += 1
            
            # Get target sentence
            if i < len(lines) and 'TGT:' in lines[i]:
                tgt_line = lines[i].strip()
                tgt_sentence = tgt_line.split('TGT:', 1)[1].strip()
                tgt_sentences.append(tgt_sentence)
        
        i += 1
    
    # Construct output filenames
    en_output = f'aligned_outputs/en_{name}_aligned.txt'
    fr_output = f'aligned_outputs/fr_{name}_aligned.txt'
    
    # Write source sentences
    with open(en_output, 'w', encoding='utf-8') as f:
        for sentence in src_sentences:
            f.write(sentence + '\n')
    
    # Write target sentences
    with open(fr_output, 'w', encoding='utf-8') as f:
        for sentence in tgt_sentences:
            f.write(sentence + '\n')
    
    print(f"Extracted {len(src_sentences)} source sentences to {en_output}")
    print(f"Extracted {len(tgt_sentences)} target sentences to {fr_output}")
    
    if len(src_sentences) != len(tgt_sentences):
        print(f"WARNING: Mismatch in sentence counts! Source: {len(src_sentences)}, Target: {len(tgt_sentences)}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python extract_aligned_files.py <name>")
        print("Example: python extract_aligned_files.py jean")
        sys.exit(1)
    
    name = sys.argv[1]
    extract_aligned_sentences(name)