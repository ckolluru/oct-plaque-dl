import os 
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(42)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

from keras.models import *
from keras.layers import GaussianNoise, Input, merge, Conv2D, MaxPooling2D, MaxPooling1D, AveragePooling1D, Dropout, Conv1D, Flatten, Dense, BatchNormalization, Activation, LocallyConnected1D, Reshape, Concatenate, Cropping1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from data import *
from generate_2D_patches import *
import pickle
import scipy.io
from keras.utils.layer_utils import print_summary
from keras import regularizers
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout
import gc

K.set_session(sess)

# Perform classification (CNN on 2D patches)
class myOCTModel(object):

    def __init__(self, aline_depth = 200, result_dir = "/home/cxk340/OCT_DL/results_cnn_patches_2D"):

        # Number of pixels in an A-line
        self.aline_depth = aline_depth

        # Results directory
        self.result_dir = result_dir
        
        # Kernel size of the first convolutional layer
        self.first_kernel_size = 11

        # Patch size (number of rows in patch)
        self.patch_size = 21


    def load_patch_data(self, kfold):

        # Load training, validation and test dataset of a particular fold in a cross validation scheme
        patch_gen = myPatchGenerator()
        patches_train, alines_labels_train, patches_validation, alines_labels_validation, patches_test, alines_labels_test = patch_gen.get_2D_patches(kfold - 1, self.patch_size)
        return patches_train, alines_labels_train, patches_validation, alines_labels_validation, patches_test, alines_labels_test

    def load_patch_data_single_train_set(self):

        # Load training, validation and test dataset of a particular fold in a cross validation scheme
        patch_gen = myPatchGenerator()
        patches_train, alines_labels_train, patches_validation, alines_labels_validation = patch_gen.get_2D_patches_single_training_set(self.patch_size)
        return patches_train, alines_labels_train, patches_validation, alines_labels_validation

    def load_data(self, kfold):

        # Load training, validation and test dataset of a particular fold in a cross validation scheme
        mydata = dataProcess()
        alines_train, alines_labels_train, alines_validation, alines_labels_validation, alines_test, alines_labels_test = mydata.load_kfold_data(kfold)
        return alines_train, alines_labels_train, alines_validation, alines_labels_validation, alines_test, alines_labels_test

    def load_ml_dataset(self):

        # Machine learning dataset to compare performance with David Prabhu's work
        mydata = dataProcess()
        patches_train, alines_labels_train, patches_validation, alines_labels_validation = mydata.load_ml_data()
        return patches_train, alines_labels_train, patches_validation, alines_labels_validation

    def expand_data(self, alines, extra_points_to_add):

        # Pads input matrix on either side by replicating the edge pixel, useful in the case of TCFAs and CNN
        on_each_side = extra_points_to_add/2
        alines_new = np.zeros((alines.shape[0], alines.shape[1], alines.shape[2] + extra_points_to_add))

        repeat_start_mat = alines[:,:,0]
        repeat_start_mat = np.expand_dims(repeat_start_mat, axis = 2)
        repeat_stop_mat = alines[:,:,-1]
        repeat_stop_mat = np.expand_dims(repeat_stop_mat, axis = 2)

        alines_new[:,:, :on_each_side] = np.repeat(repeat_start_mat, on_each_side, axis= 2)
        alines_new[:,:, -on_each_side:] = np.repeat(repeat_stop_mat, on_each_side, axis= 2)

        alines_new[:,:, on_each_side:-on_each_side] = alines[:, :, :]
        return alines_new

    def get_model(self):

        # Model architecture for A-line classification with CNN
        model = Sequential()
        model.add(Conv2D(32, (self.patch_size, self.first_kernel_size), padding='valid', activation = 'relu', kernel_initializer = 'he_normal', input_shape=(self.patch_size, self.aline_depth + self.first_kernel_size - 1, 1)))
        model.add(Reshape((-1, 32)))
        model.add(MaxPooling1D(pool_size = 2, strides = 2, padding = 'valid'))
        model.add(BatchNormalization())
        model.add(Reshape((-1,32)))
        model.add(Conv1D(64, self.first_kernel_size - 2, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(MaxPooling1D(pool_size = 2, strides = 2, padding = 'valid'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(100, kernel_initializer = 'he_normal', activation = 'relu'))
        model.add(Dense(3, kernel_initializer = 'he_normal', activation = 'softmax'))

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        print_summary(model)
        return model

    def kfoldtrain(self):

        num_folds = 10

        C = np.zeros((4,4,num_folds))
        C_percent = np.zeros((3,3,num_folds))
        
        # Perform 10 fold cross validation
        for kfold in np.arange(10):

            print('--------------Fold %d----------------' %(kfold+1))

            print('Load train, validation and test datasets')
            patches_train, alines_labels_train, patches_validation, alines_labels_validation, patches_test, alines_labels_test = self.load_patch_data(kfold+1)
            print('Load train, validation and  test datasets done')

            print('Load train, validation and test datasets')
            alines_train, alines_labels_train, alines_validation, alines_labels_validation, alines_test, alines_labels_test = self.load_data(kfold+1)
            print('Load train, validation and  test datasets done')

            print('Find which A-lines are guidewire, used to remove from confusion matrix results')
            std_alines_test_2 = np.std(alines_test, axis = 1)
            std_alines_test_2 = np.expand_dims(std_alines_test_2, axis = 1)
            std_alines_test_2 = np.repeat(std_alines_test_2, alines_test.shape[1], axis = 1)

            print('Shape of std_alines_test_2:')
            print(std_alines_test_2.shape)

            alines_train = None
            alines_validation = None
            alines_test = None

            gc.collect()

            class_weight_values = class_weight.compute_class_weight('balanced', np.unique(np.argmax(alines_labels_train,axis=1)), np.argmax(alines_labels_train, axis=1))
            class_weight_dict = dict(enumerate(class_weight_values))

            print('Class frequencies')
            print('Calcium:') 
            print(np.sum(alines_labels_train[:,0]))
            print('Lipid: ') 
            print(np.sum(alines_labels_train[:,1]))
            print('Other: ') 
            print(np.sum(alines_labels_train[:,2])) 

            print('Class weight dictionary')
            print(class_weight_dict)

            print('Expand datasets')
            patches_train = self.expand_data(patches_train, self.first_kernel_size - 1)
            patches_validation = self.expand_data(patches_validation, self.first_kernel_size - 1)
            patches_test = self.expand_data(patches_test, self.first_kernel_size - 1)
            print('Expand datasets done')

            # print('Normalize dataset (Feature-wise)')
            # mean_patches_train = np.mean(patches_train, axis = 0)            
            # std_patches_train = np.std(patches_train, axis = 0)
            # mean_patches_train = np.expand_dims(mean_patches_train, axis = 0)
            # std_patches_train = np.expand_dims(std_patches_train, axis = 0)
            # mean_patches_train = np.repeat(mean_patches_train, patches_train.shape[0], axis = 0)
            # std_patches_train = np.repeat(std_patches_train, patches_train.shape[0], axis = 0)

            # mean_patches_validation = np.mean(patches_validation, axis = 0)
            # std_patches_validation = np.std(patches_validation, axis = 0)
            # mean_patches_validation = np.expand_dims(mean_patches_validation, axis = 0)
            # std_patches_validation = np.expand_dims(std_patches_validation, axis = 0)
            # mean_patches_validation = np.repeat(mean_patches_validation, patches_validation.shape[0], axis = 0)
            # std_patches_validation = np.repeat(std_patches_validation, patches_validation.shape[0], axis = 0)

            # mean_patches_test = np.mean(patches_test, axis = 0)
            # std_patches_test = np.std(patches_test, axis = 0)
            # mean_patches_test = np.expand_dims(mean_patches_test, axis = 0)
            # std_patches_test = np.expand_dims(std_patches_test, axis = 0)
            # mean_patches_test = np.repeat(mean_patches_test, patches_test.shape[0], axis = 0)
            # std_patches_test = np.repeat(std_patches_test, patches_test.shape[0], axis = 0)

            # # Remove guidewire regions
            # for example in np.arange(patches_train.shape[0]):
            #     if std_patches_train[example, 0] != 0:
            #         patches_train[example] = np.true_divide(patches_train[example] - mean_patches_train[example], std_patches_train[example])

            # for example in np.arange(patches_validation.shape[0]):
            #     if std_patches_validation[example, 0] != 0:
            #         patches_validation[example] = np.true_divide(patches_validation[example] - mean_patches_validation[example], std_patches_validation[example])

            # for example in np.arange(patches_test.shape[0]):
            #     if std_patches_test[example, 0] != 0:
            #         patches_test[example] = np.true_divide(patches_test[example] - mean_patches_test[example], std_patches_test[example])

            print('patches_train shape: %d %d %d' %patches_train.shape)
            print('alines_labels_train shape: %d %d' %alines_labels_train.shape)

            print('patches_validation shape: %d %d %d' %patches_validation.shape)
            print('alines_labels_validation shape: %d %d' %alines_labels_validation.shape)

            print('patches_test shape: %d %d %d' %patches_test.shape)
            print('alines_labels_test shape: %d %d' %alines_labels_test.shape)

            model = self.get_model()
            print('Got OCT Model')

            early_stopping = EarlyStopping(monitor='val_loss', min_delta = 1e-4, patience = 5, verbose= 0, mode='auto')
            model_checkpoint = ModelCheckpoint(self.result_dir + '/weights_files/OCT model weights.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
            model_checkpoint_individual_fold = ModelCheckpoint(self.result_dir + '/weights_files/OCT model weights_' + str(kfold + 1) + '.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
            
            patches_train = patches_train.reshape(patches_train.shape[0], patches_train.shape[1], self.aline_depth + self.first_kernel_size - 1, 1)
            patches_validation = patches_validation.reshape(patches_validation.shape[0], patches_validation.shape[1], self.aline_depth + self.first_kernel_size - 1, 1)

            print('Fitting model...')
            history = model.fit(patches_train, alines_labels_train, batch_size=256, epochs=100, verbose=1,validation_data=(patches_validation, alines_labels_validation), shuffle=True, callbacks=[model_checkpoint, early_stopping, model_checkpoint_individual_fold], class_weight = class_weight_dict)
            with open(self.result_dir + "/history_files/history_fold" + int2str(kfold) + ".pkl", 'w') as f:
                pickle.dump(history.history, f)

            print('Predict test data')
            model.load_weights(self.result_dir + '/weights_files/OCT model weights.hdf5')
            patches_test = patches_test.reshape(patches_test.shape[0], patches_test.shape[1], self.aline_depth + self.first_kernel_size - 1, 1)
            patches_test_prediction = model.predict(patches_test, batch_size= 256, verbose = 1)

            print('Saving test data')        
            scipy.io.savemat(self.result_dir + "/Predictions Folds/alines_test_prediction_fold_" + str(kfold+1) + ".mat", mdict={'alines_test_prediction': patches_test_prediction})

            print('Computing test metrics')
            actual = 5 * np.ones(alines_labels_test.shape[0])
            predict = 5 * np.ones(alines_labels_test.shape[0]) 

            for k in np.arange(alines_labels_test.shape[0]):
                if alines_labels_test[k, 0] == 1:
                    actual[k] = 1
                if alines_labels_test[k, 1] == 1:
                    actual[k] = 2
                if alines_labels_test[k, 2] == 1:
                    actual[k] = 3
                if std_alines_test_2[k, 0] == 0:
                    actual[k] = 0

            for k in np.arange(alines_labels_test.shape[0]):
                if patches_test_prediction[k,0] >= patches_test_prediction[k,1]:
                    if patches_test_prediction[k,0] >= patches_test_prediction[k,2]:
                        predict[k] = 1

                if patches_test_prediction[k,1] >= patches_test_prediction[k,0]:
                    if patches_test_prediction[k,1] >= patches_test_prediction[k,2]:
                        predict[k] = 2

                if patches_test_prediction[k,2] >= patches_test_prediction[k,0]:
                    if patches_test_prediction[k,2] >= patches_test_prediction[k,1]:
                        predict[k] = 3

                if std_alines_test_2[k, 0] == 0:
                    predict[k] = 0

            if np.array_equal(np.unique(actual), np.unique(predict)) == False:
                print('Predictions and actual do not have the same classes')
                print('unique in actual: ')
                print(np.unique(actual))
                print('unique in predict: ')
                print(np.unique(predict))


            C[:,:,kfold] = confusion_matrix(actual, predict)
            print('Confusion Matrix')
            print(C[:,:,kfold])

            scipy.io.savemat(self.result_dir + "/Predictions Folds/Confusion_Matrix_KFold.mat", mdict={'Confusion_Matrix_All_Folds':C})

            # Remove guidewire row and column
            C_Without_Guidewire = C[1:,1:,kfold]

            for k  in np.arange(C_Without_Guidewire.shape[0]):
                C_percent[k,:,kfold] = np.true_divide(C_Without_Guidewire[k,:],np.sum(C_Without_Guidewire[k,:]))

            print('Confusion matrix in percentage')
            print(C_percent[:,:,kfold]*100)

            patches_train = None
            alines_labels_train = None
            patches_validation = None
            alines_labels_validation = None
            patches_test = None
            alines_labels_test = None

            gc.collect()

        # Remove guidewire row and column
        C_GW_Removed = C[1:,1:,:]
        C_full = np.sum(C_GW_Removed, axis = 2)
        C_full_percent = np.zeros((3,3))
        print('Confusion matrix over all folds')
        print(C_full)

        print('Confusion matrix in percentage over all folds')

        for k in np.arange(C_GW_Removed.shape[0]):
            C_full_percent[k,:] = C_full[k,:]/np.sum(C_full[k,:])

        print(C_full_percent * 100)

        print('Mean confusion matrix percentage over all folds')
        print(np.mean(C_percent*100, axis=2))

        print('Standard error of confusion matrix percentage over all folds')
        print(np.std(C_percent*100, axis=2)/np.sqrt(num_folds))

    def cryotest(self):

        C = np.zeros((4,4))
        C_percent = np.zeros((3,3))

        print('Load train, validation datasets')
        patches_train, alines_labels_train, patches_validation, alines_labels_validation = self.load_patch_data_single_train_set()
        print('Load train, validation datasets done')

        class_weight_values = class_weight.compute_class_weight('balanced', np.unique(np.argmax(alines_labels_train,axis=1)), np.argmax(alines_labels_train, axis=1))
        class_weight_dict = dict(enumerate(class_weight_values))

        print('Class frequencies')
        print('Calcium:') 
        print(np.sum(alines_labels_train[:,0]))
        print('Lipid: ') 
        print(np.sum(alines_labels_train[:,1]))
        print('Other: ') 
        print(np.sum(alines_labels_train[:,2])) 

        print('Class weight dictionary')
        print(class_weight_dict)

        print('Expand datasets')
        patches_train = self.expand_data(patches_train, self.first_kernel_size - 1)
        patches_validation = self.expand_data(patches_validation, self.first_kernel_size - 1)
        print('Expand datasets done')

        print('Normalize dataset (Feature-wise)')
        mean_patches_train = np.mean(patches_train, axis = 0)            
        std_patches_train = np.std(patches_train, axis = 0)
        mean_patches_train = np.expand_dims(mean_patches_train, axis = 0)
        std_patches_train = np.expand_dims(std_patches_train, axis = 0)
        # mean_patches_train = np.repeat(mean_patches_train, patches_train.shape[0], axis = 0)
        # std_patches_train = np.repeat(std_patches_train, patches_train.shape[0], axis = 0)

        mean_patches_validation = np.mean(patches_validation, axis = 0)
        std_patches_validation = np.std(patches_validation, axis = 0)
        mean_patches_validation = np.expand_dims(mean_patches_validation, axis = 0)
        std_patches_validation = np.expand_dims(std_patches_validation, axis = 0)
        # mean_patches_validation = np.repeat(mean_patches_validation, patches_validation.shape[0], axis = 0)
        # std_patches_validation = np.repeat(std_patches_validation, patches_validation.shape[0], axis = 0)

        print('Shape of mean patches')
        print(mean_patches_validation.shape)
        print('Shape of std patches')
        print(std_patches_validation.shape)

        # Remove guidewire regions
        for example in np.arange(patches_train.shape[0]):
                patches_train[example] = np.true_divide(patches_train[example,:,:] - mean_patches_train[:,:,:], std_patches_train[:,:,:])

        for example in np.arange(patches_validation.shape[0]):
                patches_validation[example] = np.true_divide(patches_validation[example,:,:] - mean_patches_validation[:,:,:], std_patches_validation[:,:,:])

        print('patches_train shape: %d %d %d' %patches_train.shape)
        print('alines_labels_train shape: %d %d' %alines_labels_train.shape)

        print('patches_validation shape: %d %d %d' %patches_validation.shape)
        print('alines_labels_validation shape: %d %d' %alines_labels_validation.shape)

        model = self.get_model()
        print('Got OCT Model')

        model_checkpoint = ModelCheckpoint(self.result_dir + '/weights_files/OCT model weights.hdf5')
       
        patches_train = patches_train.reshape(patches_train.shape[0], patches_train.shape[1], self.aline_depth + self.first_kernel_size - 1, 1)

        print('Fitting model...')
        history = model.fit(patches_train, alines_labels_train, batch_size=256, epochs=10, verbose=1, shuffle=True, callbacks=[model_checkpoint], class_weight = class_weight_dict)
        with open(self.result_dir + "/history_files/history.pkl", 'w') as f:
            pickle.dump(history.history, f)

        print('Predict validation data')
        model.load_weights(self.result_dir + '/weights_files/OCT model weights.hdf5')
        patches_validation = patches_validation.reshape(patches_validation.shape[0], patches_validation.shape[1], self.aline_depth + self.first_kernel_size - 1, 1)
        patches_validation_prediction = model.predict(patches_validation, batch_size= 256, verbose = 1)

        print('Saving test data')        
        scipy.io.savemat("/home/cxk340/OCT_DL/results_split_sample_validation/Predictions/alines_validation_prediction.mat", mdict={'alines_validation_prediction': patches_validation_prediction})

        print('Computing test metrics')
        actual = 5 * np.ones(alines_labels_validation.shape[0])
        predict = 5 * np.ones(alines_labels_validation.shape[0]) 

        for k in np.arange(alines_labels_validation.shape[0]):
            if alines_labels_validation[k, 0] == 1:
                actual[k] = 1
            if alines_labels_validation[k, 1] == 1:
                actual[k] = 2
            if alines_labels_validation[k, 2] == 1:
                actual[k] = 3

        for k in np.arange(alines_labels_validation.shape[0]):
            if patches_validation_prediction[k,0] >= patches_validation_prediction[k,1]:
                if patches_validation_prediction[k,0] >= patches_validation_prediction[k,2]:
                    predict[k] = 1

            if patches_validation_prediction[k,1] >= patches_validation_prediction[k,0]:
                if patches_validation_prediction[k,1] >= patches_validation_prediction[k,2]:
                    predict[k] = 2

            if patches_validation_prediction[k,2] >= patches_validation_prediction[k,0]:
                if patches_validation_prediction[k,2] >= patches_validation_prediction[k,1]:
                    predict[k] = 3

        if np.array_equal(np.unique(actual), np.unique(predict)) == False:
            print('Predictions and actual do not have the same classes')
            print('unique in actual: ')
            print(np.unique(actual))
            print('unique in predict: ')
            print(np.unique(predict))


        C[:,:] = confusion_matrix(actual, predict)
        print('Confusion Matrix')
        print(C[:,:])

        scipy.io.savemat("/home/cxk340/OCT_DL/results_split_sample_validation/Predictions/Confusion_Matrix_Cryo.mat", mdict={'Confusion_Matrix_All_Folds':C})

        # Remove guidewire row and column
        C_Without_Guidewire = C[1:,1:]

        for k  in np.arange(C_Without_Guidewire.shape[0]):
            C_percent[k,:] = np.true_divide(C_Without_Guidewire[k,:],np.sum(C_Without_Guidewire[k,:]))

        print('Confusion matrix in percentage')
        print(C_percent[:,:]*100)

        patches_train = None
        alines_labels_train = None
        patches_validation = None
        alines_labels_validation = None

        gc.collect()

        # Remove guidewire row and column
        C_GW_Removed = C[1:,1:]
        C_full = np.sum(C_GW_Removed, axis = 2)
        C_full_percent = np.zeros((3,3))
        print('Confusion matrix over all folds')
        print(C_full)

        print('Confusion matrix in percentage over all folds')

        for k in np.arange(C_GW_Removed.shape[0]):
            C_full_percent[k,:] = C_full[k,:]/np.sum(C_full[k,:])

        print(C_full_percent * 100)


if __name__ == '__main__':

    octmodel = myOCTModel(200)
    # octmodel.kfoldtrain()

    octmodel.cryotest()
