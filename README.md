Deep Learning techniques for IVOCT A-line classification

Matlab part does the pre- and post- processing, Python part has the deep neural network implementation
Matlab codes can be run on any PC with MATLAB installed on it (tested with R2016a).
Python codes need keras and tensorflow libraries installed. (tested with a singularity image on CWRU HPC)
Alternate singularity image was used to run the dense conditional random field processing.
(Some effort needed to merge into one image).

Steps to run the code:

1. Run fast_pixel_shift.m: Generates the pixel shifted images and stores them in Converted Labels folder (Using 48 in-vivo pullbacks, transform dataset).

2. Make label files (also stored in Converted Labels folder)

3. Run make_mat_arrays_each_pullbacks.m: Makes a MAT file containing A-line values and label, one for each pullback, stored in Pullbacks_Mat_Files_All_Alines

4. Run create_cross_validation_groups.m: Makes training, validation, test groups for each iteration of 10 fold cross validation procedure. (stored in Ten_Folds_All_Alines)

5. Copy over these matrices into HPC (to folder data/folds/ and train or test or validation appropriately).

6. Run oct_model.py or oct_model_ann.py if you want to run the CNN or ANN respectively. Results are stored in results_cnn/results_ann if doing the ten fold cross validation. If doing a split sample validation on a completely held out test set, change lines in these files to do that (comment line 510 in oct_model.py and uncomment line 516 in oct_model.py, comment line 471 in oct_model_ann.py, uncomment line 477 in oct_model_ann.py). Results of the split sample validation are stored in results_split_sample_validation folder

7. Copy over the test prediction matrices (Predictions Folds in the particular results folder for cross validation or Predictions in results_split_sample_validation folder) back into the PC with MATLAB on it.

8. Run visualize_predictions_en_face_view.m: Makes individual prediction matrices for each pullback

9. Copy over files in Predictions_En_Face_View to the HPC (same folder name)

10. Run noise_cancellation_crf.py 'cnn' or 'ann': This performs the CRF segment wise on each segment of the pullback

11. Copy over Predictions_CRF_Noise_Cleaned from the HPC back to the PC with MATLAB on it (into the same folder)

12. Run noise_cleaning_on_en_face_view_crf.m to get the final confusion matrices

13. If you want to view the predictions in XY domain, run visualize_predictions_xy.m

14. If you want to run the same thing but with the cryo dataset as the validation set, the validation matrix can be made by running create_split_sample_validation_set_cryo.m

15. To make predictions for each pullback in cryo dataset, run do_noise_cleaning_cryo.m till line 59. and then copy over pullback en face views to HPC and run python noise_cancellation_CRF_cryo.m cryo, and copy over to the PC with MATLAB on it. Run do_noise_cleaning_cryo again but this time the full file. This will yield confusion matrix on the cryo dataset and the predictions in XY view.

