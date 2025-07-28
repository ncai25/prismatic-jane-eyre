from lib.IBM1 import IBM1
from lib.util import write_list, read_list, draw_weighted_alignment, plot_aer, save_word_pairs
import numpy as np

def main():

    ibm = IBM1()

    english_path = 'jane-eyre/french/fr_combined_aligned.e'
    french_path = 'jane-eyre/french/fr_combined_aligned.f'

    ibm.read_data(english_path, french_path, null=True, UNK=True, max_sents=np.inf, test_repr=False)

    Save = True

    T = 20  

    for step in range(T):
        print('Iteration {}'.format(step + 1))

        save_path = 'jane-eyre/likelihoods/IBM1/EM/'
        model_path = 'jane-eyre/models/IBM1/EM/{0}-'.format(step + 1)
        word_pair_path = 'jane-eyre/word_pairs/IBM1/'

        ibm.epoch(log=True)

        if Save:
            # save translation probabilities
            ibm.save_t(model_path)
            
            # save word pairs with sentence indices
            stats = save_word_pairs(ibm, word_pair_path, step + 1)
            print(f"Word pairs saved for epoch {step+1}: {stats['english_words']} English words, {stats['unique_pairs']} unique pairs, {stats['total_occurrences']} total occurrences")

    if Save:
        write_list(ibm.likelihoods, save_path + 'likelihoods')
        ibm.plot_likelihoods(save_path + 'log-likelihood.pdf')
        write_list(ibm.null_generations, save_path + 'NULL-generations')

if __name__ == "__main__":
    main()