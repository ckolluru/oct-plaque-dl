import numpy as np
import pydensecrf.densecrf as dcrf
import scipy.io
import glob
import os
import sys

# Usage:
# noise_cancellation_crf 'cryo'
# noise_cancellation_crf 'ml' (future)
# Code supports single validation set data

# Separate directories for each classification type
cryo_directory = 'results_split_sample_validation/cryo/'

if len(sys.argv) == 2 and str(sys.argv[1]) == 'cryo':
    classification_dir = cryo_directory

# Load prediction image, each pixel has three probabilities, one for each class
list_of_mat_files_to_clean = glob.glob("/home/cxk340/OCT_DL/" + classification_dir + "Predictions Cryo En Face View/*Predictions_En_Face_View.mat")

# Avoid log(0)
epsilon = 1e-10

for file in list_of_mat_files_to_clean:

    # Get full filename and remove the Predicitions_En_Face_View.mat part, so that now we are left with short strings
    just_filename = os.path.basename(file)
    current_pullback = just_filename[:-29]

    # Load the predictions, these files are prepared by MATLAB's visualize_predictions_en_face.m
    Predictions = scipy.io.loadmat(file)
    predictions_full_pullback = Predictions['predict_reshape']

    # Define the CRF, we have three classes in this case
    d = dcrf.DenseCRF2D(predictions_full_pullback.shape[1], predictions_full_pullback.shape[0], 3)

    # Unary potentials
    U = predictions_full_pullback.transpose(2,0,1).reshape((3, -1))

    # Take negative logarithm since these are probabilities
    d.setUnaryEnergy(-np.log(U + epsilon))

    # Add pairwise Gaussian term
    d.addPairwiseGaussian(sxy=(5,19), compat=3, kernel=dcrf.DIAG_KERNEL, normalization = dcrf.NORMALIZE_SYMMETRIC)

    # Inference
    Q = d.inference(5)

    # Find class with max. probability and reshape to original shape
    map = np.argmax(Q, axis=0).reshape((predictions_full_pullback.shape[0], predictions_full_pullback.shape[1]))

    crf_results = np.array(map)

    # Save full pullback results
    scipy.io.savemat("/home/cxk340/OCT_DL/" + classification_dir + "Predictions_CRF_Noise_Cleaned/" + os.path.basename(file), mdict={'CRF_Results': crf_results})