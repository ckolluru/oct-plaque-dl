from oct_model import myOCTModel
from data import *
import scipy.io
from keras.models import *
from vis.visualization import visualize_saliency, overlay, visualize_cam
from vis.utils import utils
from keras import activations
import numpy as np
import matplotlib.cm as cm
import time
import sys
import os.path

octmodel = myOCTModel(200)
model = octmodel.get_model()

# Load weights of the fold in which TRF-04 is a test pullback
# Check if all a-lines in fold have 496 A-lines (this one does)
current_fold = 6 
# frames_in_this_fold = [28, 144, 14, 304, 101]

# Pullbacks in this fold
# 'TRF-23-0M-2-LAD-PRE-Prox': 283:310
# 'TRF-35-0M-LAD-1-PRE': 52:70, 122:141, 315:387, 459:490
# 'TRF-69-0M-1-RCA-PRE': 336:349
# 'TRF-04-0M-1-RCA-PRE': 97:242, 281:350, 413:500
# 'TRF-26-0M-1-LAD-PRE-Dist': 77:124, 153:205

# # Type of pullback (calcium or lipid or normal) Pullback and frame number
# filter_index = 0
# frame_number = 300
# pullback = 'TRF_04'

# # A-line start and stop (TRF_04_300)
# start_aline = (28+144+14+242-97+1+300-281)* 496
# stop_aline = start_aline + 496

# # Type of pullback (calcium or lipid or normal) Pullback and frame number
# filter_index = 1
# frame_number = 301
# pullback = 'TRF_23_Prox'

# # A-line start and stop (TRF_23_Prox_301)
# start_aline = (18)* 496
# stop_aline = start_aline + 496

# Type of pullback (calcium or lipid or normal) Pullback and frame number
filter_index = 1
frame_number = 159
pullback = 'TRF_26_Dist'

# A-line start and stop (TRF_26_Dist_159)
start_aline = (28+144+14+304+53)* 496
stop_aline = start_aline + 496

print(sys.argv)

if len(sys.argv) == 2 and str(sys.argv[1]) == '1':
	start_aline_partial = start_aline
	stop_aline_partial = start_aline + 248

if len(sys.argv) == 2 and str(sys.argv[1]) == '2':
	start_aline_partial = start_aline + 249
	stop_aline_partial = stop_aline

# Get string
if filter_index == 0:
    plaque_type = 'Calcium'
if filter_index == 1:
    plaque_type = 'Lipid'
if filter_index == 2:
    plaque_type = 'Other'

# String name
string_name = plaque_type + '_' + pullback + '_' + str(frame_number)

model.load_weights('/home/cxk340/OCT_DL/results_cnn/weights_files/OCT model weights_' + str(current_fold) + '.hdf5')

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'dense_2')

# Swap softmax with linear
model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)

print("Load train and test datasets")
alines_train, alines_labels_train, alines_validation, alines_labels_validation, alines_test, alines_labels_test  = octmodel.load_data(current_fold)
print("Load train and test datasets done")

print('Expand datasets')
alines_train = octmodel.expand_data(alines_train, octmodel.first_kernel_size - 1)
alines_validation = octmodel.expand_data(alines_validation, octmodel.first_kernel_size - 1)
alines_test = octmodel.expand_data(alines_test, octmodel.first_kernel_size - 1)
print('Expand datasets done')

# print('Normalize dataset (Sample-wise)') 
# mean_alines_train = np.mean(alines_train, axis = 1)
# std_alines_train = np.std(alines_train, axis = 1)
# mean_alines_train = np.expand_dims(mean_alines_train, axis = 1)
# std_alines_train = np.expand_dims(std_alines_train, axis = 1)
# mean_alines_train = np.repeat(mean_alines_train, alines_train.shape[1], axis = 1)
# std_alines_train = np.repeat(std_alines_train, alines_train.shape[1], axis = 1)

# mean_alines_validation = np.mean(alines_validation, axis = 1)
# std_alines_validation = np.std(alines_validation, axis = 1)
# mean_alines_validation = np.expand_dims(mean_alines_validation, axis = 1)
# std_alines_validation = np.expand_dims(std_alines_validation, axis = 1)
# mean_alines_validation = np.repeat(mean_alines_validation, alines_validation.shape[1], axis = 1)
# std_alines_validation = np.repeat(std_alines_validation, alines_validation.shape[1], axis = 1)

# mean_alines_test = np.mean(alines_test, axis = 1)
# std_alines_test = np.std(alines_test, axis = 1)
# mean_alines_test = np.expand_dims(mean_alines_test, axis = 1)
# std_alines_test = np.expand_dims(std_alines_test, axis = 1)
# mean_alines_test = np.repeat(mean_alines_test, alines_test.shape[1], axis = 1)
# std_alines_test = np.repeat(std_alines_test, alines_test.shape[1], axis = 1) 

# # Remove guidewire regions
# for example in np.arange(alines_train.shape[0]):
#     if std_alines_train[example, 0] != 0:
#         alines_train[example] = np.true_divide(alines_train[example] - mean_alines_train[example], std_alines_train[example])

# for example in np.arange(alines_validation.shape[0]):
#     if std_alines_validation[example, 0] != 0:
#         alines_validation[example] = np.true_divide(alines_validation[example] - mean_alines_validation[example], std_alines_validation[example])

# for example in np.arange(alines_test.shape[0]):
#     if std_alines_test[example, 0] != 0:
#         alines_test[example] = np.true_divide(alines_test[example] - mean_alines_test[example], std_alines_test[example])
                    
alines_test = alines_test.reshape(alines_test.shape[0], octmodel.aline_depth + octmodel.first_kernel_size - 1, 1)
alines_to_sample = np.arange(start_aline_partial,stop_aline_partial) 

if os.path.exists(octmodel.result_dir + "/Saliency Maps/Saliency_Map_" + string_name +".mat"):
	Grad_Full_Matrix = scipy.io.loadmat(octmodel.result_dir + "/Saliency Maps/Saliency_Map_" + string_name +".mat")
	grad_full = Grad_Full_Matrix['Saliency_Map']
else:
	grad_full = np.zeros((496, 210))

for modifier in ['guided']:
    print(modifier)

    for k in alines_to_sample:    
        current_aline  = alines_test[k,:,:]
        current_aline = np.expand_dims(current_aline, axis = 0)
        start = time.clock()
        grads = visualize_saliency(model, layer_idx, filter_indices=filter_index, seed_input=current_aline, backprop_modifier=modifier)
        print('Processed A-line %d' %(k-start_aline))
        print(grads.shape)
        grad_full[k-start_aline,:] = np.expand_dims(grads, axis = 0)    
 
    scipy.io.savemat(octmodel.result_dir + "/Saliency Maps/Saliency_Map_" + string_name +".mat", mdict={'Saliency_Map': grad_full})
    scipy.io.savemat(octmodel.result_dir + "/Saliency Maps/Original_Image_" + string_name +".mat", mdict={'ALines_Prediction':alines_test[start_aline:stop_aline,:,:]})


