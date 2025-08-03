#!/usr/bin/env python3
"""
1. Join lines broken in the middle of sentences
2. Split lines at proper sentence boundaries
"""

import re
import sys
import os

def fix_pdf_text(text):
    """
    Returns:
        Cleaned text with proper line breaks
    """
    
    # Step 1: Join lines that were broken in the middle of sentences
    # Pattern: any character that doesn't end a sentence, followed by newline
    # Replace with: the character + space
    join_pattern = r'([^.!?"\';\«»-])\n'
    text = re.sub(join_pattern, r'\1 ', text)
    hyphen_pattern = r'-\n'
    text = re.sub(hyphen_pattern, '-', text)

    # text = re.sub(r'- ', '-', text)
    text = re.sub(r'-- ', '--', text)



    # em_dash_pattern = r'--\n'
    # text = re.sub(em_dash_pattern, '--', text)
    
    # Step 2: Split at proper sentence boundaries
    # Pattern: two non-period characters, followed by sentence-ending punctuation + space
    # This avoids splitting on things like "A.B." or "..."
    split_pattern = r'([^.][^.])([.!?;:]) (?!")'
    text = re.sub(split_pattern, r'\1\2\n', text)
    text = re.sub(r'-- ', '--\n', text)
    
    closing_quote_pattern = r'\n\s*»'
    text = re.sub(closing_quote_pattern, ' »', text)

    closing_quote_pattern = r'\n\s*”'
    text = re.sub(closing_quote_pattern, ' ”', text)

    abbreviations = ["Mr", "Mrs", "Ms", "Dr", "Jr", "Sr", "St", "Prof", 
                     "Capt", "Lt", "Col", "Gen", "Sgt", "Mt", "M", "Matt", "v", "Rev", "Ch", "Cor"]
    
    # Fix each abbreviation that was incorrectly split
    for abbr in abbreviations:
        # Pattern: abbreviation followed by period and newline
        # Replace with: abbreviation, period, and space
        abbrev_pattern = rf'({abbr}\.)\n'
        text = re.sub(abbrev_pattern, r'\1 ', text)
    
    # Handle ellipsis that might get split
    ellipsis_pattern = r'\.{2,}\n'
    text = re.sub(ellipsis_pattern, r'... ', text)
    
    # Remove footnote numbers from words (Jésus14 → Jésus)
    number_suffix_pattern = r'\b(\S+?)\d+\b'
    text = re.sub(number_suffix_pattern, r'\1', text)
    
    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n+', '\n', text) 
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)

    
    return text

def process_file(input_file, output_file=None):
    """Process a single file."""
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return False
    
    # Read input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(input_file, 'r', encoding='latin1') as f:
            text = f.read()
    
    # Fix the text
    print(f"Processing {input_file}...")
    original_lines = len(text.split('\n'))
    
    fixed_text = fix_pdf_text(text)
    
    fixed_lines = len(fixed_text.split('\n'))
    print(f"  Lines: {original_lines} -> {fixed_lines}")
    
    # Write output
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_seg{ext}"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fixed_text)
    
    print(f"  Output: {output_file}")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_pdf_text.py input_file [output_file]")
        print("       python fix_pdf_text.py --batch directory/")
        sys.exit(1)
    
    if sys.argv[1] == '--batch':
        # Batch process all txt files in directory
        if len(sys.argv) < 3:
            directory = '.'
        else:
            directory = sys.argv[2]
        
        txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        
        if not txt_files:
            print(f"No .txt files found in {directory}")
            return
        
        print(f"Processing {len(txt_files)} files in {directory}...")
        
        for txt_file in txt_files:
            input_path = os.path.join(directory, txt_file)
            process_file(input_path)
            
    else:
        # Process single file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        process_file(input_file, output_file)

if __name__ == "__main__":
    main()
