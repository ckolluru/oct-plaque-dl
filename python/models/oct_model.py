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

K.set_session(sess)

# This performs classification (CNN)

class myOCTModel(object):

    def __init__(self, aline_depth = 200, result_dir = "/home/cxk340/OCT_DL/results_cnn"):

        # Number of pixels in an A-line
        self.aline_depth = aline_depth

        # Results directory
        self.result_dir = result_dir
        
        # Kernel size of the first convolutional layer
        self.first_kernel_size = 11

    def load_data(self, kfold):

        # Load training, validation and test dataset of a particular fold in a cross validation scheme
        mydata = dataProcess()
        alines_train, alines_labels_train, alines_validation, alines_labels_validation, alines_test, alines_labels_test = mydata.load_kfold_data(kfold)
        return alines_train, alines_labels_train, alines_validation, alines_labels_validation, alines_test, alines_labels_test

    def load_ml_dataset(self):

        # Machine learning dataset to compare performance with David Prabhu's work
        self.result_dir = "/home/cxk340/OCT_DL/results_ml"
        mydata = dataProcess()
        alines_train, alines_labels_train, alines_validation, alines_labels_validation = mydata.load_ml_data()
        return alines_train, alines_labels_train, alines_validation, alines_labels_validation

    def load_split_sample_dataset(self):

    	self.result_dir = "/home/cxk340/OCT_DL/results_split_sample_validation"
    	mydata = dataProcess()
    	alines_train, alines_labels_train, alines_validation, alines_labels_validation = mydata.load_split_sample_data()
    	return alines_train, alines_labels_train, alines_validation, alines_labels_validation

    def expand_data(self, alines, extra_points_to_add):

        # Pads input matrix on either side by replicating the edge pixel, useful in the case of TCFAs and CNN
        on_each_side = extra_points_to_add/2
        alines_new = np.zeros((alines.shape[0], alines.shape[1] + extra_points_to_add))

        repeat_start_mat = alines[:,0]
        repeat_start_mat = np.expand_dims(repeat_start_mat, axis = 1)
        repeat_stop_mat = alines[:,-1]
        repeat_stop_mat = np.expand_dims(repeat_stop_mat, axis = 1)

        alines_new[:, :on_each_side] = np.repeat(repeat_start_mat, on_each_side, axis= 1)
        alines_new[:, -on_each_side:] = np.repeat(repeat_stop_mat, on_each_side, axis= 1)

        alines_new[:, on_each_side:-on_each_side] = alines[:, :]
        return alines_new

    def get_model(self):

        # Model architecture for A-line classification with CNN
        model = Sequential()
        model.add(Conv1D(32, self.first_kernel_size, padding='valid', activation = 'relu', kernel_initializer = 'he_normal', input_shape=(self.aline_depth + self.first_kernel_size - 1, 1)))
        model.add(MaxPooling1D(pool_size = 2, strides = 2, padding = 'valid'))
        model.add(Reshape((-1,32)))
        model.add(Conv1D(64, self.first_kernel_size - 2, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal'))
        model.add(MaxPooling1D(pool_size = 2, strides = 2, padding = 'valid'))
        model.add(Flatten())
        model.add(Dense(100, kernel_initializer = 'he_normal', activation = 'relu'))
        model.add(Dense(3, kernel_initializer = 'he_normal', activation = 'softmax'))

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        print_summary(model)
        return model

    def kfoldtrain(self):

        num_folds = 10

    	print('Performing 10 fold internal cross validation')
    	# Extra column and row in confusion matrix for guidewire region, guidewire A-lines are removed from results
        C = np.zeros((4,4,num_folds))
        C_percent = np.zeros((3,3,num_folds))
        
        # Perform 10 fold cross validation
        for kfold in np.arange(num_folds):

            print('--------------Fold %d----------------' %(kfold+1))

            print('Load train, validation and test datasets')
            alines_train, alines_labels_train, alines_validation, alines_labels_validation, alines_test, alines_labels_test = self.load_data(kfold+1)
            print('Load train, validation and  test datasets done')

            print('Find which A-lines are guidewire, used to remove from confusion matrix results')
            std_alines_test_2 = np.std(alines_test, axis = 1)
            std_alines_test_2 = np.expand_dims(std_alines_test_2, axis = 1)
            std_alines_test_2 = np.repeat(std_alines_test_2, alines_test.shape[1], axis = 1)

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
            alines_train = self.expand_data(alines_train, self.first_kernel_size - 1)
            alines_validation = self.expand_data(alines_validation, self.first_kernel_size - 1)
            alines_test = self.expand_data(alines_test, self.first_kernel_size - 1)
            print('Expand datasets done')

            # print('Normalize dataset (Feature-wise)')
            # mean_alines_train = np.mean(alines_train, axis = 0)            
            # std_alines_train = np.std(alines_train, axis = 0)
            # mean_alines_train = np.expand_dims(mean_alines_train, axis = 0)
            # std_alines_train = np.expand_dims(std_alines_train, axis = 0)
            # mean_alines_train = np.repeat(mean_alines_train, alines_train.shape[0], axis = 0)
            # std_alines_train = np.repeat(std_alines_train, alines_train.shape[0], axis = 0)

            # mean_alines_validation = np.mean(alines_validation, axis = 0)
            # std_alines_validation = np.std(alines_validation, axis = 0)
            # mean_alines_validation = np.expand_dims(mean_alines_validation, axis = 0)
            # std_alines_validation = np.expand_dims(std_alines_validation, axis = 0)
            # mean_alines_validation = np.repeat(mean_alines_validation, alines_validation.shape[0], axis = 0)
            # std_alines_validation = np.repeat(std_alines_validation, alines_validation.shape[0], axis = 0)

            # mean_alines_test = np.mean(alines_test, axis = 0)
            # std_alines_test = np.std(alines_test, axis = 0)
            # mean_alines_test = np.expand_dims(mean_alines_test, axis = 0)
            # std_alines_test = np.expand_dims(std_alines_test, axis = 0)
            # mean_alines_test = np.repeat(mean_alines_test, alines_test.shape[0], axis = 0)
            # std_alines_test = np.repeat(std_alines_test, alines_test.shape[0], axis = 0)

            # for example in np.arange(alines_train.shape[0]):
            #         alines_train[example] = np.true_divide(alines_train[example] - mean_alines_train[example], std_alines_train[example])

            # for example in np.arange(alines_validation.shape[0]):
            #         alines_validation[example] = np.true_divide(alines_validation[example] - mean_alines_validation[example], std_alines_validation[example])

            # for example in np.arange(alines_test.shape[0]):
            #         alines_test[example] = np.true_divide(alines_test[example] - mean_alines_test[example], std_alines_test[example])

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

            print('alines_train shape: %d %d' %alines_train.shape)
            print('alines_labels_train shape: %d %d' %alines_labels_train.shape)

            print('alines_validation shape: %d %d' %alines_validation.shape)
            print('alines_labels_validation shape: %d %d' %alines_labels_validation.shape)

            print('alines_test shape: %d %d' %alines_test.shape)
            print('alines_labels_test shape: %d %d' %alines_labels_test.shape)

            model = self.get_model()
            # plot_model(model, to_file='/home/cxk340/OCT_DL/models/model.png')
            print('Got OCT Model')

            early_stopping = EarlyStopping(monitor='val_loss', min_delta = 1e-4, patience = 5, verbose= 0, mode='auto')
            model_checkpoint = ModelCheckpoint(self.result_dir + '/weights_files/OCT model weights.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
            model_checkpoint_individual_fold = ModelCheckpoint(self.result_dir + '/weights_files/OCT model weights_' + str(kfold + 1) + '.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
            
            alines_train = alines_train.reshape(alines_train.shape[0], self.aline_depth + self.first_kernel_size - 1, 1)
            alines_validation = alines_validation.reshape(alines_validation.shape[0], self.aline_depth + self.first_kernel_size - 1, 1)

            print('Fitting model...')
            history = model.fit(alines_train, alines_labels_train, batch_size=256, epochs=100, verbose=1,validation_data=(alines_validation, alines_labels_validation), shuffle=True, callbacks=[model_checkpoint, early_stopping, model_checkpoint_individual_fold], class_weight = class_weight_dict)
            with open(self.result_dir + "/history_files/history_fold_" + str(kfold+1) + ".pkl", 'w') as f:
                pickle.dump(history.history, f)

            print('Predict test data')
            model.load_weights(self.result_dir + '/weights_files/OCT model weights.hdf5')
            alines_test = alines_test.reshape(alines_test.shape[0], self.aline_depth + self.first_kernel_size - 1, 1)
            alines_test_prediction = model.predict(alines_test, batch_size= 256, verbose = 1)

            print('Saving test data')        
            scipy.io.savemat(self.result_dir + "/Predictions Folds/alines_test_prediction_fold_" + str(kfold+1) + ".mat", mdict={'alines_test_prediction': alines_test_prediction})

            print('Computing test metrics')
            actual = 5 * np.ones(alines_labels_test.shape[0])
            predict = 5 * np.ones(alines_labels_test.shape[0]) 

            print('Find which A-lines are guidewire, used to remove from confusion matrix results')
            mean_alines_test = np.mean(alines_test, axis = 1)
            std_alines_test = np.std(alines_test, axis = 1)
            mean_alines_test = np.expand_dims(mean_alines_test, axis = 1)
            std_alines_test = np.expand_dims(std_alines_test, axis = 1)
            mean_alines_test = np.repeat(mean_alines_test, alines_test.shape[1], axis = 1)
            std_alines_test = np.repeat(std_alines_test, alines_test.shape[1], axis = 1)

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
                if alines_test_prediction[k,0] >= alines_test_prediction[k,1]:
                    if alines_test_prediction[k,0] >= alines_test_prediction[k,2]:
                        predict[k] = 1

                if alines_test_prediction[k,1] >= alines_test_prediction[k,0]:
                    if alines_test_prediction[k,1] >= alines_test_prediction[k,2]:
                        predict[k] = 2

                if alines_test_prediction[k,2] >= alines_test_prediction[k,0]:
                    if alines_test_prediction[k,2] >= alines_test_prediction[k,1]:
                        predict[k] = 3

                if std_alines_test_2[k, 0] == 0:
                    predict[k] = 0

            if np.array_equal(np.unique(actual), np.unique(predict)) == False:
                print('Predictions and actual do not have the same classes')
                print('unique in actual: ')
                print(np.unique(actual))
                print('unique in predict: ')
                print(np.unique(predict))
            else:
            	print('Classes in predict and actual')
            	print(np.unique(actual))


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


    def train_with_validation_set(self, train_ml = True, train_split_sample_validation = False):

        print('Load train, validation datasets')

        if train_ml and not train_split_sample_validation:
        	print('Performing single training and validation on ML dataset')
        	alines_train, alines_labels_train, alines_validation, alines_labels_validation= self.load_ml_dataset()
        if not train_ml and train_split_sample_validation:
        	print('Performing single training and validation on split sample validation dataset')
        	alines_train, alines_labels_train, alines_validation, alines_labels_validation= self.load_split_sample_dataset()
        if not train_ml and not train_split_sample_validation:
        	print('Illegal call for train_with_validation_set. Exiting..')
        if train_ml and train_split_sample_validation:
        	print('Illegal call for train_with_validation_set. Exiting..')

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

        print('Find which A-lines are guidewire, used to remove from confusion matrix results')
        std_alines_validation_2 = np.std(alines_validation, axis = 1)
        std_alines_validation_2 = np.expand_dims(std_alines_validation_2, axis = 1)
        std_alines_validation_2 = np.repeat(std_alines_validation_2, alines_validation.shape[1], axis = 1)

        print('Normalize dataset (Feature-wise)')
        mean_alines_train = np.mean(alines_train, axis = 0)
        std_alines_train = np.std(alines_train, axis = 0)
        mean_alines_train = np.expand_dims(mean_alines_train, axis = 0)
        std_alines_train = np.expand_dims(std_alines_train, axis = 0)
        mean_alines_train = np.repeat(mean_alines_train, alines_train.shape[0], axis = 0)
        std_alines_train = np.repeat(std_alines_train, alines_train.shape[0], axis = 0)

        mean_alines_validation = np.mean(alines_validation, axis = 0)
        std_alines_validation = np.std(alines_validation, axis = 0)
        mean_alines_validation = np.expand_dims(mean_alines_validation, axis = 0)
        std_alines_validation = np.expand_dims(std_alines_validation, axis = 0)
        mean_alines_validation = np.repeat(mean_alines_validation, alines_validation.shape[0], axis = 0)
        std_alines_validation = np.repeat(std_alines_validation, alines_validation.shape[0], axis = 0)

        for example in np.arange(alines_train.shape[0]):
                alines_train[example] = np.true_divide(alines_train[example] - mean_alines_train[example], std_alines_train[example])

        for example in np.arange(alines_validation.shape[0]):
                alines_validation[example] = np.true_divide(alines_validation[example] - mean_alines_validation[example], std_alines_validation[example])

        print('Expand datasets')
        alines_train = self.expand_data(alines_train, self.first_kernel_size - 1)
        alines_validation = self.expand_data(alines_validation, self.first_kernel_size - 1)
        print('Expand datasets done')

        print('alines_train shape: %d %d' %alines_train.shape)
        print('alines_labels_train shape: %d %d' %alines_labels_train.shape)

        print('alines_validation shape: %d %d' %alines_validation.shape)
        print('alines_labels_validation shape: %d %d' %alines_labels_validation.shape)

        model = self.get_model()
        print('Got OCT Model')
        
        alines_train = alines_train.reshape(alines_train.shape[0], self.aline_depth + self.first_kernel_size - 1, 1)
        alines_validation = alines_validation.reshape(alines_validation.shape[0], self.aline_depth + self.first_kernel_size - 1, 1)

        print('Fitting model...')
        history = model.fit(alines_train, alines_labels_train, batch_size=256, epochs=10, verbose=1, shuffle=True, class_weight = class_weight_dict)

        model.save(self.result_dir + "/weights_files/OCT model weights.hdf5")

        with open(self.result_dir + "/history_files/history.pkl", 'w') as f:
            pickle.dump(history.history, f)

        print('Predict validation data')
        model.load_weights(self.result_dir + '/weights_files/OCT model weights.hdf5')
        alines_validation_prediction = model.predict(alines_validation, batch_size= 256, verbose = 1)

        print('Saving validation data')        
        scipy.io.savemat(self.result_dir + "/Predictions/alines_validation_prediction.mat", mdict={'alines_validation_prediction': alines_validation_prediction})

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
            if std_alines_validation_2[k, 0] == 0:
            	actual[k] = 0


        for k in np.arange(alines_labels_validation.shape[0]):
            if alines_validation_prediction[k,0] >= alines_validation_prediction[k,1]:
                if alines_validation_prediction[k,0] >= alines_validation_prediction[k,2]:
                    predict[k] = 1

            if alines_validation_prediction[k,1] >= alines_validation_prediction[k,0]:
                if alines_validation_prediction[k,1] >= alines_validation_prediction[k,2]:
                    predict[k] = 2

            if alines_validation_prediction[k,2] >= alines_validation_prediction[k,0]:
                if alines_validation_prediction[k,2] >= alines_validation_prediction[k,1]:
                    predict[k] = 3

            if std_alines_validation_2[k,0] == 0:
            	predict[k] = 0

        if np.array_equal(np.unique(actual), np.unique(predict)) == False:
            print('Predictions and actual do not have the same classes')
            print('unique in actual: ')
            print(np.unique(actual))
            print('unique in predict: ')
            print(np.unique(predict))
        else:
        	print('Classes in unique and actual')
        	print(np.unique(actual))

        # Three classes for fibrocalcific, fibrolipdiic and other. One extra class for guidewire which 
        # will not be counted in the final confusion matrix
        C = np.zeros((4,4))
        C_percent = np.zeros((3,3))

        C[:,:] = confusion_matrix(actual, predict)

        # Delete first row and first column because that corresponds to guidewire
        C = np.delete(C, (0), axis = 0)
        C = np.delete(C, (0), axis = 1)

        print('Confusion Matrix')
        print(C[:,:])

        scipy.io.savemat(self.result_dir + "/Predictions/Confusion_Matrix_ML.mat", mdict={'Confusion_Matrix_ML':C})

        for k  in np.arange(C.shape[0]):
            C_percent[k,:] = np.true_divide(C[k,:],np.sum(C[k,:]))

        print('Confusion matrix in percentage')
        print(C_percent[:,:]*100)

if __name__ == '__main__':

    octmodel = myOCTModel(200)
    
    # Perform 10 fold cross validation
    octmodel.kfoldtrain()

    # Perform ML experiment
    #octmodel.train_with_validation_set(True, False)

    # Perform split sample validation experiment
    # octmodel.train_with_validation_set(False, True)
