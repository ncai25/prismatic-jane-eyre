from lib.IBM2 import IBM2
from lib.util import write_list, read_list, draw_weighted_alignment, plot_aer, plot_jump, save_word_pairs
from lib.aer_import import test
import matplotlib.pyplot as plt
import numpy as np

def main():

	ibm = IBM2()

	english_path = 'jane-eyre/french/fr_combined_aligned.e'
	french_path = 'jane-eyre/french/fr_combined_aligned.f'

	ibm.read_data(english_path, french_path, null=True,  UNK=True, max_sents=np.inf, random_init=False, test_repr=False)
	ibm.load_t('jane-eyre/models/IBM1/EM/20-')

	print(np.sum(ibm.t))

	Save = True

	T = 15

	for step in range(T):
		
		print('Iteration {}'.format(step+1))

		save_path 		= 'jane-eyre/likelihoods/IBM2/pretrained-init/'
		model_path 		= 'jane-eyre/models/IBM2/pretrained-init/{0}-'.format(step+1)
		word_pair_path = 'jane-eyre/word_pairs/IBM2/'
		
		ibm.epoch(log=True)
		if Save:		
			# save translation probabilities
			ibm.save_t(model_path)

			# save jump probabilities
			ibm.save_jump(model_path)
			
			# save word pairs with sentence indices
			stats = save_word_pairs(ibm, word_pair_path, step + 1)
			print(f"Word pairs saved for epoch {step+1}: {stats['english_words']} English words, {stats['unique_pairs']} unique pairs, {stats['total_occurrences']} total occurrences")

	if Save:
		write_list(ibm.likelihoods, save_path + 'likelihoods')
		ibm.plot_likelihoods(save_path + 'log-likelihood.pdf')
		plot_jump(ibm.jump, ibm.max_jump, save_path)


if __name__ == "__main__":
	main()
