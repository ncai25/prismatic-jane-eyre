from lib.IBM1 import IBM1
from lib.util import write_list, save_word_pairs
import numpy as np

def main():

    ibm = IBM1()

    english_path = 'preprocess/tokenized/en_tokenized_all.txt'
    french_path = 'preprocess/tokenized/fr_tokenized_all.txt'

    ibm.read_data(english_path, french_path, null=True, UNK=True, max_sents=np.inf, test_repr=False)

    Save = True

    T = 20  

    for step in range(T):
        print('Iteration {}'.format(step + 1))

        save_path = 'ibm_model/likelihoods/IBM1/EM/'
        model_path = 'ibm_model/models/IBM1/EM/{0}-'.format(step + 1)
        word_pair_path = 'ibm_model/word_pairs/IBM1/'

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

if __name__ == "__main__":
    main()