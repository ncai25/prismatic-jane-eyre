from lib.IBM1 import IBM1
from lib.util import write_list, read_list, draw_weighted_alignment, plot_aer
from lib.aer_import import test
import matplotlib.pyplot as plt
import numpy as np


def main():
    ibm = IBM1()

    english_path = 'jane-eyre/Fr_all.e'
    french_path = 'jane-eyre/Fr_all.f'

    ibm.read_data(english_path, french_path, null=True, UNK=True, max_sents=np.inf, test_repr=False)

    Save = True

    T = 14  # Number of training iterations
    aers = []

    for step in range(T):
        print('Iteration {}'.format(step + 1))

        # Setting saving paths
        save_path = 'prediction/validation/IBM1/EM/'
        model_path = 'jane-eyre/models/IBM1/EM/{0}-'.format(step + 1)
        alignment_path = save_path + 'prediction-{0}'.format(step + 1)

        # Run EM step
        ibm.epoch(log=True)

        # Predict alignments
        ibm.predict_alignment('validation/dev.f',
                              'validation/dev.e',
                              alignment_path)

        # Calculate AER
        aer = test('validation/dev.wa.nonullalign',
                   alignment_path)

        aers.append(aer)
        print('AER: {}'.format(aer))
        print('Total NULL alignments: {}'.format(ibm.null_generations[-1]))

        if Save:
            # Save translation probabilities
            ibm.save_t(model_path)

    if Save:
        # Save likelihoods
        write_list(ibm.likelihoods, save_path + 'likelihoods')
        ibm.plot_likelihoods(save_path + 'log-likelihood.pdf')

        # Save AERs
        write_list(aers, save_path + 'AERs')
        plot_aer(aers, save_path)

        # Save total NULL alignments
        write_list(ibm.null_generations, save_path + 'NULL-generations')

    # ✅ FINAL STEP: Extract Final Alignment Pairs After Training
    # final_alignment_output = save_path + 'final_alignment_pairs.txt'

    # with open(final_alignment_output, 'w') as out_file:
    #     for F, E in zip(ibm.french, ibm.english):
    #         _, word_pairs = ibm.align(F, E, return_pairs=True)  # Extract word pairs
    #         for f_word, e_word in word_pairs:
    #             out_file.write(f"{f_word} --> {e_word}\n")
    #         out_file.write("\n")  # Separate sentences

    # print(f"Final alignment pairs saved to {final_alignment_output}")

if __name__ == "__main__":
    main()