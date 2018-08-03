from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import scipy.io
import h5py

# Processes data from MAT files and makes them available as numpy arrays
# Can load particular folds in a cross validation dataset or a singular dataset
class dataProcess(object):

    def __init__(self, train_path = "/home/cxk340/OCT_DL/data/folds/train/",
                       validation_path = "/home/cxk340/OCT_DL/data/folds/validation/",
                       test_path = "/home/cxk340/OCT_DL/data/folds/test/",
                       ml_train_path = "/home/cxk340/OCT_DL/data/ml_dataset/train/",
                       ml_validation_path = "/home/cxk340/OCT_DL/data/ml_dataset/validation/",
                       split_sample_train_path = "/home/cxk340/OCT_DL/data/split_sample_validation/train/",
                       split_sample_validation_path = "/home/cxk340/OCT_DL/data/split_sample_validation/validation/"):

        self.train_path = train_path
        self.validation_path = validation_path
        self.test_path = test_path

        self.ml_train_path = ml_train_path
        self.ml_validation_path = ml_validation_path

        self.split_sample_train_path = split_sample_train_path
        self.split_sample_validation_path = split_sample_validation_path

    def load_kfold_data(self, kfold, log_scaling = True):
        print('-' * 30)
        print('Load K-fold dataset')
        print('-' * 30)

        # Load training dataset
        f = h5py.File(self.train_path + "Fold" + str(kfold) + "_Train.mat")
        ALine_Label_Training_Matrix = {}

        for k,v in f.items():
        	ALine_Label_Training_Matrix[k] = np.array(v).transpose()

        training_data = ALine_Label_Training_Matrix['ALine_Label_Training_Matrix']

        alines_train = training_data[:, :-3]
        alines_labels_train = training_data[:,-3:]

        # Load validation dataset
        ALine_Label_Validation_Matrix = scipy.io.loadmat(self.validation_path + "Fold" + str(kfold) + "_Validation.mat")
        validation_data = ALine_Label_Validation_Matrix['ALine_Label_Validation_Matrix']

        alines_validation = validation_data[:,:-3]
        alines_labels_validation = validation_data[:,-3:]

        # Load test dataset
        ALine_Label_Test_Matrix = scipy.io.loadmat(self.test_path + "Fold" + str(kfold) + "_Test.mat")
        testing_data = ALine_Label_Test_Matrix['ALine_Label_Test_Matrix']

        alines_test = testing_data[:,:-3]
        alines_labels_test = testing_data[:,-3:]

        # Perform log scaling
        if log_scaling:
        	alines_train = np.log(alines_train + 1)
        	alines_validation = np.log(alines_validation + 1)
        	alines_test = np.log(alines_test + 1)

        return alines_train, alines_labels_train, alines_validation, alines_labels_validation, alines_test, alines_labels_test

    def load_ml_data(self, log_scaling = True):

    	# Load training dataset (full)
        f = h5py.File(self.ml_train_path + "Training_for_ML_Full_Dataset.mat")
        ALine_Label_Training_Matrix = {}

        for k,v in f.items():
        	ALine_Label_Training_Matrix[k] = np.array(v).transpose()

        # # Load training dataset (partial)
        # ALine_Label_Training_Matrix = scipy.io.loadmat(self.ml_train_path + "Training_for_ML.mat")


        training_data = ALine_Label_Training_Matrix['ALine_Label_Matrix']

        alines_train = training_data[:,:-3]
        alines_labels_train = training_data[:,-3:]

        # Load held-out validation dataset
        ALine_Label_Validation_Matrix = scipy.io.loadmat(self.ml_validation_path + "Validation_for_ML.mat")
        validation_data = ALine_Label_Validation_Matrix['ALine_Label_Matrix']

        alines_validation = validation_data[:,:-3]
        alines_labels_validation = validation_data[:,-3:]

        # Perform log scaling
        if log_scaling:
        	alines_train = np.log(alines_train + 1)
        	alines_validation = np.log(alines_validation + 1)

        return alines_train, alines_labels_train, alines_validation, alines_labels_validation

    def load_split_sample_data(self, log_scaling = True):

        # Load training dataset (full or full+2cryo)
        f = h5py.File(self.split_sample_train_path + "Training_Matrix_All_48_Pullbacks.mat")
        # f = h5py.File(self.split_sample_train_path + "Training_Matrix_All_48_Pullbacks_Plus_2_Cryo_Pullbacks.mat")
        ALine_Label_Training_Matrix = {}

        for k,v in f.items():
            ALine_Label_Training_Matrix[k] = np.array(v).transpose()

        # # Load training dataset (partial)
        # ALine_Label_Training_Matrix = scipy.io.loadmat(self.ml_train_path + "Training_for_ML.mat")


        training_data = ALine_Label_Training_Matrix['ALine_Label_Matrix']

        alines_train = training_data[:,:-3]
        alines_labels_train = training_data[:,-3:]

        # Load held-out validation dataset (either Split_Sample_Validation or Split_Sample_Cryo)
        ALine_Label_Validation_Matrix = scipy.io.loadmat(self.split_sample_validation_path + "Split_Sample_Cryo.mat")
        validation_data = ALine_Label_Validation_Matrix['ALine_Label_Matrix']

        alines_validation = validation_data[:,:-3]
        alines_labels_validation = validation_data[:,-3:]

        # Perform log scaling
        if log_scaling:
            alines_train = np.log(alines_train + 1)
            alines_validation = np.log(alines_validation + 1)

        print('Training A-lines shape: %d %dataset' %alines_train.shape)
        print('Validation A-lines shape: %d %d' %alines_validation.shape)

        return alines_train, alines_labels_train, alines_validation, alines_labels_validation

if __name__ == "__main__":

    mydata = dataProcess()