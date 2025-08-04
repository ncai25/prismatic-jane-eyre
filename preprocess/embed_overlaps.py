from sentence_transformers import SentenceTransformer 
import os
import numpy as np

model = SentenceTransformer('sentence-transformers/LaBSE', device='mps')

for filename in os.listdir('overlaps'):
    if filename.endswith('_overlap'):
        input_path = f'overlaps/{filename}'
        output_name = filename.replace('_overlap', '.emb')
        output_path = f'embed/{output_name}'
        
        print(f'Processing {filename}...')
        with open(input_path, 'r') as f:
            sentences = f.readlines()
        embeddings = model.encode(sentences, show_progress_bar=True)
        embeddings.astype(np.float32).tofile(output_path)