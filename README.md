# Prismatic Jane Eyre - IBM Translation Models

This project applies IBM Models 1 and 2 to analyze translations of Charlotte Brontë's _Jane Eyre_, tracing how key words and phrases evolve across different French and Italian translations from various time periods (1854-2014). The goal is to provide computational support for literary translation analysis using statistical word alignment models.

## Data

The Jane Eyre translation corpus includes:

- Multiple French translations (1854-2008)
- Multiple Italian translations (1904-2014)
- Aligned sentence pairs between English source and translations

## Implementation

### Sentence Alignment with Vecalign

This project uses [vecalign](https://github.com/thompsonb/vecalign) during preprocessing for high-quality sentence alignment between source and target texts. Vecalign leverages multilingual sentence embeddings (LaBSE) to achieve better alignment quality before applying IBM Models for word-level analysis.

### IBM Models

This implementation builds upon code from [IBM-Models by daandouwe](https://github.com/daandouwe/IBM-Models), originally developed for the UvA NLP2 course. The models have been adapted for literary text analysis, featuring IBM Model 1 and 2 implementation with EM parameter estimation.

## Usage

### Preprocessing Pipeline

#### 1. Sentence Segmentation

Normalize texts and ensure consistent sentence segmentation across different translation versions:

```bash
python sent_segmenter.py --batch seg_texts/
```

#### 2. Generate Overlapping Sentences

Create overlapping sentence combinations for alignment:

```bash
for file in seg_texts/*.txt; do
    base=$(basename "$file" .txt)
    ./vecalign/overlap.py -i "$file" -o "overlaps/${base}_overlap.txt" -n 10
done
```

#### 3. Generate Sentence Embeddings

Create embeddings using LaBSE model:

```bash
python embed_overlaps.py
```

#### 4. Run Sentence Alignment

Align English and target language texts. For example:

```bash
vecalign/vecalign.py --alignment_max_size 8 \
    --src seg_texts/en_gutenburg.txt \
    --tgt seg_texts/fr_gilbert_duvivier.txt \
    --src_embed overlaps/en_gutenburg_overlap embed/en_gutenburg.emb \
    --tgt_embed overlaps/fr_gilbert_duvivier_overlap embed/fr_gilbert_duvivier.emb \
    --print_aligned_text > output.txt
```

#### 5. Extract Aligned Sentences

Extract aligned sentence pairs into separate files:

```bash
python extract_aligned_files.py
```

### Training IBM Models

#### IBM Model 1

```bash
python run-ibm1.py
```

#### IBM Model 2

```bash
python run-ibm2.py
```

### Analysis

Extract word contexts and analyze translations:

```bash
python extract_word_contexts.py
```

## Acknowledgments

- **IBM Model Implementation**: Based on code from [daandouwe/IBM-Models](https://github.com/daandouwe/IBM-Models)
  - Original authors: Daan van Stigt, Fije van Overeem, and Tim van Elsloo
  - Adapted for literary translation analysis
- **Sentence Alignment**: Using [vecalign](https://github.com/thompsonb/vecalign) for multilingual sentence alignment
  - Thompson, Brian and Philipp Koehn (2019): Vecalign: Improved Sentence Alignment in Linear Time and Space
  - Achieves better alignment quality using multilingual sentence embeddings (LaBSE)

## Requirements

```
pip install numpy
pip install matplotlib
pip install tabulate
pip install progressbar2
pip install sentence-transformers
```

## License

[Specify your license]

## Citation

If using this code for research, please cite:

- Original IBM Models: Brown et al. (1993)
- This implementation: [Your citation]
