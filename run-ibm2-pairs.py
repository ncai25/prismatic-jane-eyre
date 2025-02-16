from lib.IBM2 import IBM2
from lib.util import write_list, read_list, draw_weighted_alignment, plot_aer, plot_jump
from lib.aer_import import test
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

def main():

	ibm = IBM2()

	english_path = 'jane-eyre/prep-full.e'
	french_path = 'jane-eyre/prep-full.f'


	ibm.read_data(english_path, french_path, null=True,  UNK=True, max_sents=np.inf, random_init=False, test_repr=False)
	ibm.load_t('jane-eyre/models/IBM1/EM/14-')

	print(np.sum(ibm.t))

	Save = True

	T = 14
	aers = []
	for step in range(T):
		
		print('Iteration {}'.format(step+1))

		# setting saving paths
		save_path 		= 'prediction/validation/IBM2/pretrained-init/'
		model_path 		= 'jane-eyre/models/IBM2/pretrained-init/{0}-'.format(step+1)
		alignment_path 	= save_path + 'prediction-{0}'.format(step+1)
		
		ibm.epoch(log=True)

		ibm.predict_alignment('validation/dev.f', 
							  'validation/dev.e', 
							  alignment_path)
		
		aer = test('validation/dev.wa.nonullalign', 
				   alignment_path)
		
		aers.append(aer)
		print('AER: {}'.format(aer))

		# draw weighted alignments for sentence 21 (not working properly)
		draw_weighted_alignment(ibm, alignment_path,
									 'validation/dev.f', 
									 'validation/dev.e', 
									 'prediction/validation/sentence-draws/IBM1-sentence-21-iter-{}'.format(step+1), 
									 sentence=20)
		if Save:		
			# save translation probabilities
			ibm.save_t(model_path)
			# save jump probabilities
			ibm.save_jump(model_path)

	if Save:
		# save likelihoods
		write_list(ibm.likelihoods, save_path + 'likelihoods')
		# plot likelihoods
		ibm.plot_likelihoods(save_path + 'log-likelihood.pdf')
		# save aers
		write_list(aers, save_path + 'AERs')
		# plot aers
		plot_aer(aers, save_path)
		# plot jump distribution
		plot_jump(ibm.jump, ibm.max_jump, save_path)

	# ibm.tabulate_t(english_words=['the', 'and', 'me', 'is', 'where', 'of', 'or', '-NULL-'], k=4)

	# Save as both TXT and CSV
	word_pair_path_txt = save_path + 'word_pairs.txt'
	word_pair_path_csv = save_path + 'word_pairs.csv'

	os.makedirs(os.path.dirname(word_pair_path_txt), exist_ok=True)

	# Collect word pairs with probabilities
	word_pairs = []
	for f_word, f_index in ibm.V_f_indices.items():
		for e_word, e_index in ibm.V_e_indices.items():
			prob = ibm.t[f_index, e_index]
			if prob > 0.01:  # Meaningful threshold
				word_pairs.append((f_word, e_word, prob))

	# Sort word pairs by probability (descending)
	word_pairs.sort(key=lambda x: x[2], reverse=True)

	# Save as formatted TXT
	with open(word_pair_path_txt, 'w') as f:
		f.write("French_Word -> English_Word : Probability\n")
		f.write("=" * 40 + "\n")
		for f_word, e_word, prob in word_pairs:
			f.write(f"{f_word} -> {e_word} : {prob:.4f}\n")

	# Save as CSV for easier data analysis
	with open(word_pair_path_csv, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['French_Word', 'English_Word', 'Probability'])
		for f_word, e_word, prob in word_pairs:
			writer.writerow([f_word, e_word, f"{prob:.4f}"])

	print(f"\nWord pairs saved to:")
	print(f" - {word_pair_path_txt}")
	print(f" - {word_pair_path_csv}")

if __name__ == "__main__":
	main()
