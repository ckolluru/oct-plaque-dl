import scipy.io
import sklearn.feature_extraction
from data import *
import time
import gc

# Generates 2D patches based on input of numpy arrays containing all A-lines and labels (from dataProcess())
# Patch size can be specified, takes information about the pullbacks used in each fold, the frames which were annotated (Frame_Numbers.mat) and 
# number of A-lines in a frame in a particular pullback (ALines_In_Frame.mat)
class myPatchGenerator(object):

    def __init__(self, pullback_info_dir = "/home/cxk340/OCT_DL/data/folds/pullback_info/"):

        self.pullback_info_dir = pullback_info_dir

        self.training_indices = scipy.io.loadmat(self.pullback_info_dir + 'Training_Pullback_Indices_In_Each_Fold.mat')
        self.training_folds_full = self.training_indices['training_folds_full']

        self.validation_indices = scipy.io.loadmat(self.pullback_info_dir + 'Validation_Pullback_Indices_In_Each_Fold.mat')
        self.validation_folds_full = self.validation_indices['validation_folds_full']

        self.testing_indices = scipy.io.loadmat(self.pullback_info_dir + 'Test_Pullback_Indices_In_Each_Fold.mat')
        self.test_folds_full = self.testing_indices['test_folds_full']

        self.alines_in_frame_matrix = scipy.io.loadmat(self.pullback_info_dir + 'ALines_In_Frame.mat')
        self.alines_in_frame = self.alines_in_frame_matrix['alines_in_frame']
        self.alines_in_frame = self.alines_in_frame[0]

        self.frame_numbers_matrix = scipy.io.loadmat(self.pullback_info_dir + 'Frame_Numbers.mat')
        self.frame_numbers = self.frame_numbers_matrix['frame_nums']

    def get_2D_patches(self, kfold, num_alines_in_patch):

        mydata = dataProcess()

        # Load just A-lines and their labels normally from dataProcess object
        alines_train, alines_labels_train, alines_validation, alines_labels_validation, alines_test, alines_labels_test = mydata.load_kfold_data(kfold+1, False)

        # Pullback indices in training, validation and testing
        train_indices = self.training_folds_full[kfold, :]
        validation_indices = self.validation_folds_full[kfold, :]
        test_indices = self.test_folds_full[kfold, :]

        # Remove zeros from list
        np.trim_zeros(train_indices, trim='b')
        np.trim_zeros(validation_indices, trim='b')
        np.trim_zeros(test_indices, trim='b')

        # Frames in each pullback
        frames_in_each_pullback = np.zeros((self.frame_numbers.shape[0]))
        alines_in_each_pullback = np.zeros((self.frame_numbers.shape[0]))

        # Calculate number of frames and A-lines in each pullback
        for k in np.arange(self.frame_numbers.shape[0]):

            current_pullback_frames = self.frame_numbers[k]

            for segment in np.arange(current_pullback_frames.shape[0]):

                frames_in_each_pullback[k] = frames_in_each_pullback[k] + current_pullback_frames[segment].size

            alines_in_each_pullback[k] = frames_in_each_pullback[k] * self.alines_in_frame[k]

        # Generate patches for training, validation and test
        patches_train = self.generate_patches(alines_train, train_indices, alines_in_each_pullback, self.alines_in_frame, num_alines_in_patch, self.frame_numbers)
        patches_validation = self.generate_patches(alines_validation, validation_indices, alines_in_each_pullback, self.alines_in_frame, num_alines_in_patch, self.frame_numbers)
        patches_test = self.generate_patches(alines_test, test_indices, alines_in_each_pullback, self.alines_in_frame, num_alines_in_patch, self.frame_numbers)

        print('Training patches shape: %d %d %d' %patches_train.shape)
        print('Validation patches shape: %d %d %d' %patches_validation.shape)
        print('Testing patches shape: %d %d %d' %patches_test.shape)

        np.set_printoptions(threshold='nan')

        # Sanity check, this has to be zero
        print('Sum of differences %d ' %np.sum(alines_train[98654-10:98654+11, :] - patches_train[98654, :, :]))
        
        patches_train_double = np.zeros_like(patches_train)
        patches_train_double = np.log(patches_train + 1)

        patches_train = None
        gc.collect()

        patches_validation_double = np.zeros_like(patches_validation)
        patches_validation_double = np.log(patches_validation + 1)

        patches_validation = None
        gc.collect()

        patches_test_double = np.zeros_like(patches_test)
        patches_test_double = np.log(patches_test + 1)
        
        patches_test = None
        gc.collect()

        return patches_train_double, alines_labels_train, patches_validation_double, alines_labels_validation, patches_test_double, alines_labels_test

    
    def get_2D_patches_single_training_set(self, num_alines_in_patch):

        mydata = dataProcess()

        alines_train, alines_labels_train, alines_validation, alines_labels_validation = mydata.load_split_sample_data(False)
        frame_numbers_validation = scipy.io.loadmat("/home/cxk340/OCT_DL/data/split_sample_validation/validation/Frame_Numbers_Cryo.mat")
        frame_numbers_validation = frame_numbers_validation['frame_nums']

        # Frames in each pullback
        frames_in_each_pullback = np.zeros((self.frame_numbers.shape[0]))
        alines_in_each_pullback = np.zeros((self.frame_numbers.shape[0]))

        # Calculate number of frames and A-lines in each pullback
        for k in np.arange(self.frame_numbers.shape[0]):

            current_pullback_frames = self.frame_numbers[k]

            for segment in np.arange(current_pullback_frames.shape[0]):

                frames_in_each_pullback[k] = frames_in_each_pullback[k] + current_pullback_frames[segment].size

            alines_in_each_pullback[k] = frames_in_each_pullback[k] * self.alines_in_frame[k]

        
        # Frames in each pullback
        frames_in_each_pullback_validation = np.zeros((frame_numbers_validation.shape[0]))
        alines_in_each_pullback_validation = np.zeros((frame_numbers_validation.shape[0]))

        # Calculate number of frames and A-lines in each pullback
        for k in np.arange(frame_numbers_validation.shape[0]):

            current_pullback_frames = frame_numbers_validation[k]

            for segment in np.arange(current_pullback_frames.shape[0]):

                frames_in_each_pullback_validation[k] = frames_in_each_pullback_validation[k] + current_pullback_frames[segment].size

            alines_in_each_pullback_validation[k] = frames_in_each_pullback_validation[k] * 504
        
        patches_train = self.generate_patches(alines_train, np.arange(1, 49), alines_in_each_pullback, self.alines_in_frame, num_alines_in_patch, self.frame_numbers)

        print(self.alines_in_frame)
        print(np.ones((9)) * 504)
        
        patches_validation = self.generate_patches(alines_validation, np.arange(1, 10), alines_in_each_pullback_validation, np.ones((9)) * 504, num_alines_in_patch, frame_numbers_validation)

        print('Training patches shape: %d %d %d' %patches_train.shape)
        print('Validation patches shape: %d %d %d' %patches_validation.shape)

        np.set_printoptions(threshold='nan')

        # Sanity check, this has to be zero
        print('Sum of differences %d ' %np.sum(alines_train[98654-10:98654+11, :] - patches_train[98654, :, :]))
        
        patches_train_double = np.zeros_like(patches_train)
        patches_train_double = np.log(patches_train + 1)

        patches_train = None
        gc.collect()

        patches_validation_double = np.zeros_like(patches_validation)
        patches_validation_double = np.log(patches_validation + 1)

        patches_validation = None
        gc.collect()

        return patches_train_double, alines_labels_train, patches_validation_double, alines_labels_validation

    def generate_patches(self, alines, pullback_indices, alines_in_each_pullback, alines_in_frame, num_alines_in_patch, frame_numbers):

        patches_full = np.zeros((alines.shape[0], num_alines_in_patch, alines.shape[1]), dtype=np.int16)

        alines_processed = 0

        # Loop through each pullback
        for pullback in np.arange(pullback_indices.size):

            print('Pullback: %d ' %pullback)

            current_pullback_index = pullback_indices[pullback]

            if current_pullback_index == 0:
                continue

            frame_numbers_for_current_pullback = frame_numbers[current_pullback_index - 1]

            start_aline_for_current_pullback = alines_processed
            alines_processed = alines_processed + alines_in_each_pullback[current_pullback_index - 1]

            frames_processed = 0

            # Loop through each segment
            for segment in np.arange(frame_numbers_for_current_pullback.shape[0]):

                if frame_numbers_for_current_pullback[segment].shape[0] == 0:
                    continue

                current_frames_in_segment = frame_numbers_for_current_pullback[segment][0]
                start_aline_for_current_segment = start_aline_for_current_pullback + (frames_processed * alines_in_frame[current_pullback_index - 1])

                frames_processed = frames_processed + current_frames_in_segment.size

                if current_frames_in_segment.size == 0:
                    continue

                # Loop through each frame
                for frame in np.arange(current_frames_in_segment.shape[0]):
                    start_aline_for_current_frame = start_aline_for_current_segment + (frame * alines_in_frame[current_pullback_index - 1])
                    stop_aline_for_current_frame = start_aline_for_current_frame + alines_in_frame[current_pullback_index - 1] 

                    current_frame = alines[int(start_aline_for_current_frame):int(stop_aline_for_current_frame),:]

                    # Pad the frame by replication 
                    pad_on_each_side = num_alines_in_patch / 2
                    current_frame_padded = np.zeros((current_frame.shape[0] + num_alines_in_patch - 1, current_frame.shape[1]))

                    repeat_start_mat = current_frame[0,:]
                    repeat_start_mat = np.expand_dims(repeat_start_mat, axis = 0)
                    repeat_stop_mat = alines[-1,:]
                    repeat_stop_mat = np.expand_dims(repeat_stop_mat, axis = 0)

                    current_frame_padded[:pad_on_each_side, :] = np.repeat(repeat_start_mat, pad_on_each_side, axis= 0)
                    current_frame_padded[-pad_on_each_side:, :] = np.repeat(repeat_stop_mat, pad_on_each_side, axis= 0)

                    current_frame_padded[pad_on_each_side:-pad_on_each_side, :] = current_frame[:, :]

                    # Use scipy feature_extraction library's extract_patches_2d
                    patches = sklearn.feature_extraction.image.extract_patches_2d(current_frame_padded, (num_alines_in_patch, current_frame.shape[1]))

                    patches_full[int(start_aline_for_current_frame):int(stop_aline_for_current_frame), :, :] = patches

        return patches_full

if __name__ == "__main__":

    patch_gen = myPatchGenerator()

    for kfold in np.arange(10):
        patch_gen.get_2D_patches(kfold, 21)
