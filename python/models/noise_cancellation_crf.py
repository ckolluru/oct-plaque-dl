import numpy as np
import pydensecrf.densecrf as dcrf
import scipy.io
import glob
import os
import sys

# Usage:
# noise_cancellation_crf 'cnn'
# noise_cancellation_crf 'ann'
# noise_cancellation_crf 'patch'
# Code supports cross validation folds currently

# Separate directories for each classification type
cnn_directory = 'results_cnn'
ann_directory = 'results_ann'
cnn_patches_directory = 'results_cnn_patches_2D'

if len(sys.argv) == 2 and str(sys.argv[1]) == 'cnn':
    classification_dir = cnn_directory
if len(sys.argv) == 2 and str(sys.argv[1]) == 'ann':
    classification_dir = ann_directory
if len(sys.argv) == 2 and str(sys.argv[1]) == 'patch':
    classification_dir = cnn_patches_directory

# Load prediction image, each pixel has three probabilities, one for each class
list_of_mat_files_to_clean = glob.glob("/home/cxk340/OCT_DL/" + classification_dir + "/Predictions En Face View/*.mat")

# Get pullback string list
pullbackShortStringsDict = scipy.io.loadmat('/home/cxk340/OCT_DL/data/folds/pullback_info/PullbackShortStrings.mat')
pullbackShortStrings = pullbackShortStringsDict['pullbackShortStrings'].transpose()
pullbackList = [str(''.join(letter)) for letter_array in pullbackShortStrings[0] for letter in letter_array]

# Get corresponding frame numbers list
frame_numbers_dict = scipy.io.loadmat('/home/cxk340/OCT_DL/data/folds/pullback_info/Frame_Numbers.mat')
frame_numbers = frame_numbers_dict['frame_nums']

# Avoid log(0)
epsilon = 1e-10

for file in list_of_mat_files_to_clean:

    # Get full filename and remove the Predicitions_En_Face_View.mat part, so that now we are left with short strings
    just_filename = os.path.basename(file)
    current_pullback = just_filename[:-29]

    # Get the frame numbers which were labeled for this pullback, each labeled segment is one array in this list
    pullback_index = pullbackList.index(current_pullback)
    frame_indices = frame_numbers[pullback_index, :]

    # Load the predictions, these files are prepared by MATLAB's visualize_predictions_en_face.m
    Predictions = scipy.io.loadmat(file)
    predictions_full_pullback = Predictions['predict_reshape']

    # Make an empty array for this pullback that will hold final results
    crf_results = np.zeros((predictions_full_pullback.shape[0], predictions_full_pullback.shape[1]), dtype=int)

    # Keep track of how many frames were analyzed in each segment
    number_of_frames_analyzed = 0

    # Perform CRF for each segment separately
    for segment in np.arange(frame_indices.shape[0]):

        # We are only interested in the number of frames in each segment
        number_of_frames_in_segment = frame_indices[segment].shape[1]

        # If no frames are seen in a segment, stop analyzing this pullback
        if frame_indices[segment].shape[1] == 0:
            break
        else:
            # Find start and stop frame for this segment
            start_frame_for_crf = number_of_frames_analyzed
            stop_frame_for_crf = start_frame_for_crf + number_of_frames_in_segment

            predictions = predictions_full_pullback[:,start_frame_for_crf:stop_frame_for_crf, :]

            # Define the CRF, we have three classes in this case
            d = dcrf.DenseCRF2D(predictions.shape[1], predictions.shape[0], 3)

            # Unary potentials
            U = predictions.transpose(2,0,1).reshape((3, -1))

            # Take negative logarithm since these are probabilities
            d.setUnaryEnergy(-np.log(U + epsilon))

            # Add pairwise Gaussian term
            d.addPairwiseGaussian(sxy=(5, 19), compat=3, kernel=dcrf.DIAG_KERNEL, normalization = dcrf.NORMALIZE_SYMMETRIC)

            # Inference
            Q = d.inference(5)

            # Find class with max. probability and reshape to original shape
            map = np.argmax(Q, axis=0).reshape((predictions.shape[0], predictions.shape[1]))

            crf_segment_results = np.array(map)

            # Compose full pullback results from individual segments
            crf_results[:, start_frame_for_crf:stop_frame_for_crf] = crf_segment_results

        # Get start frame for next segment
        number_of_frames_analyzed = number_of_frames_analyzed + number_of_frames_in_segment

    # Save full pullback results
    scipy.io.savemat("/home/cxk340/OCT_DL/" + classification_dir + "/Predictions_CRF_Noise_Cleaned/" + os.path.basename(file), mdict={'CRF_Results': crf_results})