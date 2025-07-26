from lib.IBM2 import IBM2
from lib.util import write_list, read_list, draw_weighted_alignment, plot_aer, plot_jump
from lib.aer_import import test
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def main():

	ibm = IBM2()

	english_path = 'jane-eyre/french/Fr_1966_Monod.e'
	french_path = 'jane-eyre/french/Fr_1966_Monod.f'

	ibm.read_data(english_path, french_path, null=True,  UNK=True, max_sents=np.inf, random_init=False, test_repr=False)
	ibm.load_t('jane-eyre/models/IBM1/EM/20-')

	print(np.sum(ibm.t))

	Save = True

	T = 20

	for step in range(T):
		
		print('Iteration {}'.format(step+1))

		save_path 		= 'jane-eyre/prediction/validation/IBM2/pretrained-init/'
		os.makedirs(save_path, exist_ok=True)
		model_path 		= 'jane-eyre/models/IBM2/pretrained-init/{0}-'.format(step+1)
		os.makedirs('jane-eyre/models/IBM2/pretrained-init/', exist_ok=True)
		alignment_path 	= save_path + 'prediction-{0}'.format(step+1)
		
		ibm.epoch(log=True)

		ibm.predict_alignment('jane-eyre/french/Fr_1966_Monod.f', 
							  'jane-eyre/french/Fr_1966_Monod.e', 
							  alignment_path)
	
		if Save:		
			# save translation probabilities
			ibm.save_t(model_path)
			# save jump probabilities
			ibm.save_jump(model_path)

	if Save:
		write_list(ibm.likelihoods, save_path + 'likelihoods')
		ibm.plot_likelihoods(save_path + 'log-likelihood.pdf')
		plot_jump(ibm.jump, ibm.max_jump, save_path)

	word_pair_path_json = save_path + 'word_pairs.json'

	word_pairs = []
	for f_word, f_index in ibm.V_f_indices.items():
		for e_word, e_index in ibm.V_e_indices.items():
			prob = ibm.t[f_index, e_index]
			if prob > 0.0001:
				word_pairs.append({
					'french_word': f_word,
					'english_word': e_word,
					'probability': round(prob, 4),
				})

	word_pairs.sort(key=lambda x: x['probability'], reverse=True)
	
	with open(word_pair_path_json, 'w', encoding='utf-8') as jsonfile:
		json.dump(word_pairs, jsonfile, ensure_ascii=False, indent=2)

	print(f"\nWord pairs saved to:")
	print(f" - {word_pair_path_json}")
	print(f" - Total pairs: {len(word_pairs)}")
	print(f" - File size: {os.path.getsize(word_pair_path_json) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
	main()
