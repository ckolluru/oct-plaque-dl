import numpy as np
import scipy.io
from sklearn.metrics import roc_curve, auc
import sys
import glob
import os

import matplotlib.pyplot as plt

# Usage:
# python ROC_Curves.py 'cnn'
# python ROC_Curves.py 'ann'
# python ROC_Curves.py 'patch'

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
list_of_mat_files = glob.glob("/home/cxk340/OCT_DL/" + classification_dir + "/Predictions En Face View/*.mat")

start_aline = 0
predictions = np.zeros((2212880, 3))

for file in list_of_mat_files:

    # Get full filename and remove the Predicitions_En_Face_View.mat part, so that now we are left with short strings
    just_filename = os.path.basename(file)
    current_pullback = just_filename[:-29]
    
    # Load the predictions, these files are prepared by MATLAB's visualize_predictions_en_face.m
    Predictions = scipy.io.loadmat(file)
    predictions_full_pullback = Predictions['predict_reshape']
    predictions_full_pullback = np.reshape(predictions_full_pullback, (predictions_full_pullback.shape[0]*predictions_full_pullback.shape[1], 3))

    # Copy the predictions over to a large matrix
    stop_aline = start_aline + predictions_full_pullback.shape[0]
    predictions[start_aline:stop_aline, :] = predictions_full_pullback
    start_aline = stop_aline

print('Total A-lines processed: %d' %stop_aline)

# list_of_mat_files = glob.glob("/home/cxk340/OCT_DL/data/pullbacks/*.mat")

start_aline = 0
labels = np.zeros((2212880, 3))

for file in list_of_mat_files:

    # Get full filename and remove the Predicitions_En_Face_View.mat part, so that now we are left with short strings
    just_filename = os.path.basename(file)
    current_pullback = just_filename[:-29]

    # Load the labels
    labels_full_pullback = scipy.io.loadmat("/home/cxk340/OCT_DL/data/pullbacks/" + current_pullback + ".mat")
    labels_full_pullback = labels_full_pullback['ALine_Label_Matrix']

    labels_full_pullback = labels_full_pullback[:, -3:]

    # Copy the labels over to a large matrix
    stop_aline = start_aline + labels_full_pullback.shape[0]
    labels[start_aline:stop_aline, :] = labels_full_pullback
    start_aline = stop_aline

print('Total A-lines processed: %d' %stop_aline)

fpr_calcium, tpr_calcium, thresholds_calcium = roc_curve(labels[:,0], predictions[:,0])
roc_auc_calcium = auc(fpr_calcium, tpr_calcium)

fpr_lipid, tpr_lipid, thresholds_lipid = roc_curve(labels[:,1], predictions[:,1])
roc_auc_lipid = auc(fpr_lipid, tpr_lipid)

fpr_other, tpr_other, thresholds_other = roc_curve(labels[:,2], predictions[:,2])
roc_auc_other = auc(fpr_other, tpr_other)

optimal_idx_calc = np.argmax(tpr_calcium - fpr_calcium)

optimal_idx_lipid = np.argmax(tpr_lipid - fpr_lipid)

optimal_idx_other = np.argmax(tpr_other - fpr_other)

calc_threshold = thresholds_calcium[optimal_idx_calc]
lipid_threshold = thresholds_lipid[optimal_idx_lipid]
other_threshold = thresholds_other[optimal_idx_other]

print(calc_threshold)
print(lipid_threshold)
print(other_threshold)

# Plot ROC curve
fig = plt.figure()
plt.plot(fpr_calcium, tpr_calcium, label='Calcium (area = %0.2f)' %roc_auc_calcium, color='deeppink', linewidth=4)
plt.plot(fpr_lipid, tpr_lipid, label='Lipid (area = %0.2f)' %roc_auc_lipid, color='darkorange', linewidth=4)
plt.plot(fpr_other, tpr_other, label='Other (area = %0.2f)' %roc_auc_other, color='blue', linewidth=4)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (Validation Dataset)')
plt.legend(loc="lower right")
# plt.show()
fig.savefig('/home/cxk340/OCT_DL/' + classification_dir + '/ROC curves_Full_Dataset.png')


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
n_classes = 3
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(labels[:, i],
                                                        predictions[:, i])
    average_precision[i] = average_precision_score(labels[:,i], predictions[:,i])

from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange'])
lines = []
labels = []
plt.figure(figsize=(7, 8))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (average_precision_score = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.figure()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


plt.show()
