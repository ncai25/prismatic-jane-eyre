# Prismatic Jane Eyre - IBM Translation Models

This project applies IBM Models 1 and 2 to analyze translations of Charlotte Brontë's _Jane Eyre_ across multiple languages, focusing on how key words and phrases evolve across different translations.

## Project Overview

Using statistical word alignment models, this project examines French and Italian translations of _Jane Eyre_ to trace how specific words and their contexts change across different translators and time periods. The goal is to provide computational support for literary translation analysis.

## Data

The Jane Eyre translation corpus includes:

- Multiple French translations (1854-2008)
- Multiple Italian translations (1904-2014)
- Aligned sentence pairs between English source and translations

## Implementation

### IBM Models

This implementation builds upon code from [IBM-Models by daandouwe](https://github.com/daandouwe/IBM-Models), originally developed for the UvA NLP2 course. The models have been adapted for literary text analysis.

### Key Features

- IBM Model 1 and 2 implementation with EM parameter estimation
- Word pair extraction with probability thresholds
- Context extraction for translation analysis
- Visualization of alignment probabilities

## Usage

### Training IBM Model 1

```bash
python run-ibm1.py
```

### Training IBM Model 2

```bash
python run-ibm2.py
```

### Extracting Word Contexts

```bash
python extract_word_contexts.py
```

## Acknowledgments

- **IBM Model Implementation**: Based on code from [daandouwe/IBM-Models](https://github.com/daandouwe/IBM-Models)
  - Original authors: Daan van Stigt, Fije van Overeem, and Tim van Elsloo
  - Adapted for literary translation analysis
- **Sentence Alignment**: Using [bleualign](https://github.com/rsennrich/Bleualign) for preprocessing
  - Sennrich, Rico and Martin Volk (2010): MT-based Sentence Alignment for OCR-generated Parallel Texts

## Requirements

```
pip install numpy
pip install matplotlib
pip install tabulate
pip install progressbar2
```

## License

[Specify your license]

## Citation

If using this code for research, please cite:

- Original IBM Models: Brown et al. (1993)
- This implementation: [Your citation]
