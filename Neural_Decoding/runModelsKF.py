# Training and Testing on X-Component and Y-Component

# Import models 
import numpy as np
from numpy.linalg import inv as inv #Used in kalman filter

from Neural_Decoding.decoders import KalmanFilterDecoder
from Neural_Decoding.metrics import get_R2, get_rho, get_R2_parts

def split_dataset(curr_input, curr_output, training_range, testing_range, valid_range, num_examples):

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

    return X_train, y_train, X_test, y_test, X_valid, y_valid
    

def run_model_kf(input, output, training_range, testing_range, valid_range, type_of_R2):
   
    R2s = []
    models = []
    for i in range(len(input)):
        curr_input = input[i]
        curr_output = output[i]
        num_examples=curr_input.shape[0] # nRows (b/c nCols = number of units)

        # #Number of examples after taking into account bins removed for lag alignment
        # num_examples_kf=X_kf.shape[0]

        # Split input, output into training, testing, and validation sets 
        X_train, y_train, X_test, y_test, X_valid, y_valid = split_dataset(curr_input, curr_output, training_range, testing_range, valid_range, num_examples)

        #Declare model
        model = KalmanFilterDecoder(C=1) #There is one optional parameter (see ReadMe)

        #Fit model
        model.fit(X_train,y_train)
        
        #Save fitted models for later (i.e. cross-bucket tests)
        models.append(model)
    
        #Get predictions
        y_valid_predicted = model.predict(X_valid, y_valid)

        #Get metrics of fit (see read me for more details on the differences between metrics)
        # 1st and 2nd entries that correspond to the velocities
        if type_of_R2 == "score": # Computing single-component FVAF
            R2_kf = get_R2(y_valid, y_valid_predicted)
        elif type_of_R2 == "parts": # Can be used to later compute combined FVAF
            R2_kf = get_R2_parts(y_valid, y_valid_predicted)
        R2s.append(R2_kf)

    return R2s, models 
    

def cross_buckets_test(models, input, output, training_range, testing_range, valid_range, type_of_R2):
    """
    (SIKE) Just need X_valid and y_valid from the cross-bucket in order to predict its kinematics using model trained on the other bucket.
    """
    
    XY_FVAF = [[] for i in range(len(input))]
    Nomen = [[] for i in range(len(input))]
    Denom = [[] for i in range(len(input))]
    total_residual = []
    
    for m in range(len(models)): # but in reality, nModels should/will always be the same as nCrossBuckets
        R2s = []
        for c in range(len(input)):
            if m == c: 
                 continue 
            else: 
                curr_input = input[c]
                curr_output = output[c]
                num_examples=curr_input.shape[0] # nRows (b/c nCols = number of units)

                # Split input, output into training, testing, and validation sets 
                X_train, y_train, X_test, y_test, X_valid, y_valid = split_dataset(curr_input, curr_output, training_range, testing_range, valid_range, num_examples)
                
                #Get predictions
                y_valid_predicted = models[m].predict(X_valid, y_valid)

                #Get metrics of fit (see read me for more details on the differences between metrics)
                # 1st and 2nd entries that correspond to the velocities
                if type_of_R2 == "score": # Computing single-component FVAF
                    R2_kf = get_R2(y_valid, y_valid_predicted)
                elif type_of_R2 == "parts": # Can be used to later compute combined FVAF
                    R2_kf = get_R2_parts(y_valid, y_valid_predicted)
                    # print(R2_kf)
                    # print(R2_kf.shape)
                R2s.append(R2_kf)

                # Compute combined XY_FVAF
                vel_x_nom = R2_kf[0][0] # dim = (nom, x_vel)
                vel_x_denom = R2_kf[1][0] # dim = (denom, x_vel)
                vel_y_nom = R2_kf[0][1] # dim = (nom, y_vel)
                vel_y_denom = R2_kf[1][1] # dim = (denom, y_vel)
                nom = vel_x_nom + vel_y_nom
                denom = vel_x_denom + vel_y_denom

                combined_FVAF = 1 - (nom / denom)
                XY_FVAF[m].append(combined_FVAF)

                Nomen[m].append(nom)
                Denom[m].append(denom)

        total_residual.append(1 - (sum(Nomen[m]) / sum(Denom[m])))      

    return XY_FVAF, total_residual


def cross_polarity_test(models, input, output, training_range, testing_range, valid_range, type_of_R2, frag_type):   
    """
    """

    # Can only run this test for AD and HV fragments, NOT VM fragments 
    if frag_type == "AD" or frag_type == "HV":    

        XY_FVAF = []
        for m in range(len(models)): # but in reality, nModels should/will always be the same as nCrossBuckets
            # n is the index of the bucket with opposite polarity 
            if m <= 7:
                n = m + 8
            else:
                n = m - 8

            curr_input = input[n]
            curr_output = output[n]
            num_examples=curr_input.shape[0] # nRows (b/c nCols = number of units)

            # Split input, output into training, testing, and validation sets 
            _, _, _, _, X_valid, y_valid = split_dataset(curr_input, curr_output, training_range, testing_range, valid_range, num_examples)        
            
            #Get predictions
            y_valid_predicted = models[m].predict(X_valid, y_valid)

            #Get metrics of fit (see read me for more details on the differences between metrics)
            # 1st and 2nd entries that correspond to the velocities
            if type_of_R2 == "score": # Computing single-component FVAF
                R2_kf = get_R2(y_valid, y_valid_predicted)
            elif type_of_R2 == "parts": # Can be used to later compute combined FVAF
                R2_kf = get_R2_parts(y_valid, y_valid_predicted)

            # Compute combined XY_FVAF
            vel_x_nom = R2_kf[0][0] # dim = (nom, x_vel)
            vel_x_denom = R2_kf[1][0] # dim = (denom, x_vel)
            vel_y_nom = R2_kf[0][1] # dim = (nom, y_vel)
            vel_y_denom = R2_kf[1][1] # dim = (denom, y_vel)
            nom = vel_x_nom + vel_y_nom
            denom = vel_x_denom + vel_y_denom

            combined_FVAF = 1 - (nom / denom)
            XY_FVAF.append(combined_FVAF)

    elif frag_type == "VM":
        XY_FVAF = []

    return XY_FVAF



def complete_opposite_bucket_test(models, input, output, training_range, testing_range, valid_range, type_of_R2, frag_type):   
    """
    """
    
    XY_FVAF = []
    for m in range(len(models)): # but in reality, nModels should/will always be the same as nCrossBuckets
        
        if frag_type == "AD" or frag_type == "HV":    
            # n is the index of the bucket with opposite direction and polarity 
            # (based on the 'combos' variable which specifies bucket labels)
            if m >= 0 and m <= 3:    
                n = m + 12
            elif m >= 4 and m <= 7:
                n = m + 4
            elif m >= 8 and m <= 11:
                n = m - 4
            elif m >= 12 and m <= 15:
                n = m - 12
        
        elif frag_type == "VM":
            if m >= 0 and m <=3:
                n = m + 4
            elif m >=4 and m <=7:
                n = m - 4

        curr_input = input[n]
        curr_output = output[n]
        num_examples=curr_input.shape[0] # nRows (b/c nCols = number of units)

        # Split input, output into training, testing, and validation sets 
        _, _, _, _, X_valid, y_valid = split_dataset(curr_input, curr_output, training_range, testing_range, valid_range, num_examples)  
        
        #Get predictions
        y_valid_predicted = models[m].predict(X_valid, y_valid)

        #Get metrics of fit (see read me for more details on the differences between metrics)
        # 1st and 2nd entries that correspond to the velocities
        if type_of_R2 == "score": # Computing single-component FVAF
            R2_kf = get_R2(y_valid, y_valid_predicted)
        elif type_of_R2 == "parts": # Can be used to later compute combined FVAF
            R2_kf = get_R2_parts(y_valid, y_valid_predicted)

        # Compute combined XY_FVAF
        vel_x_nom = R2_kf[0][0] # dim = (nom, x_vel)
        vel_x_denom = R2_kf[1][0] # dim = (denom, x_vel)
        vel_y_nom = R2_kf[0][1] # dim = (nom, y_vel)
        vel_y_denom = R2_kf[1][1] # dim = (denom, y_vel)
        nom = vel_x_nom + vel_y_nom
        denom = vel_x_denom + vel_y_denom

        combined_FVAF = 1 - (nom / denom)
        XY_FVAF.append(combined_FVAF)

    return XY_FVAF
            