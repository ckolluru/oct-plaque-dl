import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

# Separate directories for each classification type
cnn_directory = 'results_cnn'
ann_directory = 'results_ann'
cnn_patches_directory = 'results_cnn_patches_2D'

if len(sys.argv) == 3 and str(sys.argv[1]) == 'cnn':
    classification_dir = cnn_directory
if len(sys.argv) == 3 and str(sys.argv[1]) == 'ann':
    classification_dir = ann_directory
if len(sys.argv) == 3 and str(sys.argv[1]) == 'patch':
    classification_dir = cnn_patches_directory

if len(sys.argv) == 3:
	fold_num = sys.argv[2]

# read from pickle file
with open('/home/cxk340/OCT_DL/' + classification_dir + '/history_files/history_fold_' + str(fold_num) + '.pkl') as f:
	history = pickle.load(f)

# list all data in history
print(history.keys())
# summarize history for accuracy
fig = plt.figure()
accuracy = history['acc']
val_accuracy = history['val_acc']

for num in range(len(accuracy)):
    accuracy[num] *= 100

for num in range(len(val_accuracy)):
    val_accuracy[num] *= 100

plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title('Plot of network accuracy')
plt.ylabel('Classification accuracy(%)')
plt.xlabel('Number of epochs')
plt.ylim((50, 100))
plt.xlim(xmin=0)
plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper right')
fig.savefig('/home/cxk340/OCT_DL/' + classification_dir + '/graphs/Accuracy vs epoch.png')

# summarize history for loss
fig2 = plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig2.savefig('/home/cxk340/OCT_DL/' + classification_dir + '/graphs/Loss vs epoch.png')

accuracy = history['acc']
val_accuracy = history['val_acc']

loss = history['loss']
val_loss = history['val_loss']

print('Max accuracy and minimum loss for all epochs')

print('Max Train Accuracy: %f ' %np.max(accuracy))
print('Max Validation Accuracy: %f' %np.max(val_accuracy))
print('Min Train Loss: %f ' %np.min(loss))
print('Min Validation Loss: %f' %np.min(val_loss))


plt.close('all')

