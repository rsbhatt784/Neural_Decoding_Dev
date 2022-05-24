# Training and Testing on X-Component and Y-Component

# Import models 
import numpy as np
from numpy.linalg import inv as inv #Used in kalman filter

from Neural_Decoding.decoders import KalmanFilterDecoder
from Neural_Decoding.metrics import get_R2, get_rho, get_R2_parts

def run_model_kf(input, output, training_range, testing_range, valid_range, type_of_R2):
   
    R2s = []
    for i in range(len(input)):
        curr_input = input[i]
        curr_output = output[i]
        num_examples=curr_input.shape[0] # nRows (b/c nCols = number of units)

        # #Number of examples after taking into account bins removed for lag alignment
        # num_examples_kf=X_kf.shape[0]
                
        #Note that each range has a buffer of 1 bin at the beginning and end
        #This makes it so that the different sets don't include overlapping data
        training_set=np.arange(int(np.round(training_range[0]*num_examples))+1,int(np.round(training_range[1]*num_examples))-1)
        testing_set=np.arange(int(np.round(testing_range[0]*num_examples))+1,int(np.round(testing_range[1]*num_examples))-1)
        valid_set=np.arange(int(np.round(valid_range[0]*num_examples))+1,int(np.round(valid_range[1]*num_examples))-1)      

        # #Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
        # #This makes it so that the different sets don't include overlapping neural data
        # training_set=np.arange(int(np.round(training_range[0]*num_examples))+bins_before,int(np.round(training_range[1]*num_examples))-bins_after)
        # testing_set=np.arange(int(np.round(testing_range[0]*num_examples))+bins_before,int(np.round(testing_range[1]*num_examples))-bins_after)
        # valid_set=np.arange(int(np.round(valid_range[0]*num_examples))+bins_before,int(np.round(valid_range[1]*num_examples))-bins_after)

        #Get training data
        X_train=curr_input[training_set,:]
        y_train=curr_output[training_set,:]

        #Get testing data
        X_test=curr_input[testing_set,:]
        y_test=curr_output[testing_set,:]

        #Get validation data
        X_valid=curr_input[valid_set,:]
        y_valid=curr_output[valid_set,:]

        #Z-score "X" inputs. 
        X_train_mean=np.nanmean(X_train,axis=0)
        X_train_std=np.nanstd(X_train,axis=0)
        X_train=(X_train-X_train_mean)/X_train_std
        X_test=(X_test-X_train_mean)/X_train_std
        X_valid=(X_valid-X_train_mean)/X_train_std

        #Zero-center outputs
        y_train_mean=np.mean(y_train,axis=0)
        y_train=y_train-y_train_mean
        y_test=y_test-y_train_mean
        y_valid=y_valid-y_train_mean

        #Declare model
        model = KalmanFilterDecoder(C=1) #There is one optional parameter (see ReadMe)

        #Fit model
        model.fit(X_train,y_train)

        #Get predictions
        y_valid_predicted = model.predict(X_valid, y_valid)

        #Get metrics of fit (see read me for more details on the differences between metrics)
        # 1st and 2nd entries that correspond to the positions
        if type_of_R2 == "score":
            R2_kf = get_R2(y_valid, y_valid_predicted)
        else:
            R2_kf = get_R2_parts(y_valid, y_valid_predicted)
        R2s.append(R2_kf)

            # #Next I'll get the rho^2 (the pearson correlation squared)
            # rho_kf=get_rho(y_valid, y_valid_predicted_kf)
            # #print('rho2:',rho_kf[0:2]**2) #I'm just printing the rho^2's of the 1st and 2nd entries that correspond to the positions
            # Rhos.append(rho_kf) 

    return R2s
