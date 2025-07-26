from lib.IBM1 import IBM1
from lib.util import write_list, read_list, draw_weighted_alignment, plot_aer
# from lib.aer_import import test
import matplotlib.pyplot as plt
import numpy as np


def main():

    ibm = IBM1()

    english_path = 'jane-eyre/french/Fr_1966_Monod.e'
    french_path = 'jane-eyre/french/Fr_1966_Monod.f'

    ibm.read_data(english_path, french_path, null=True, UNK=True, max_sents=np.inf, test_repr=False)

    Save = True

    T = 50  
    # aers = []

    for step in range(T):
        print('Iteration {}'.format(step + 1))

        # Setting saving paths
        save_path = 'jane-eyre/prediction/validation/IBM1/EM/'
        model_path = 'jane-eyre/models/IBM1/EM/{0}-'.format(step + 1)
        alignment_path = save_path + 'prediction-{0}'.format(step + 1)

        ibm.epoch(log=True)

        ibm.predict_alignment('jane-eyre/french/Fr_1966_Monod.f',
                              'jane-eyre/french/Fr_1966_Monod.e',
                              alignment_path)

        # aer = test('validation/dev.wa.nonullalign',
        #            alignment_path)

        # aers.append(aer)
        # print('AER: {}'.format(aer))
        # print('Total NULL alignments: {}'.format(ibm.null_generations[-1]))

        if Save:
            # Save translation probabilities
            ibm.save_t(model_path)

    if Save:
        # Save likelihoods
        write_list(ibm.likelihoods, save_path + 'likelihoods')
        ibm.plot_likelihoods(save_path + 'log-likelihood.pdf')

        # Save AERs
        # write_list(aers, save_path + 'AERs')
        # plot_aer(aers, save_path)

        # Save total NULL alignments
        write_list(ibm.null_generations, save_path + 'NULL-generations')

if __name__ == "__main__":
    main()