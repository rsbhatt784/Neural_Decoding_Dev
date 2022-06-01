import numpy as np

########## R-squared (R2) ##########

def get_R2(y_test,y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """

    R2_list=[] #Initialize a list that will contain the R2s for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute R2 for each output
        y_mean=np.mean(y_test[:,i])
        R2=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        R2_list.append(R2) #Append R2 of this output to the list
    # R2_array=np.array(R2_list)
    # return R2_array #Return an array of R2s
    return R2_list

########## R-squared (R2) Components (nom and denom) ##########

def get_R2_parts(y_test,y_test_pred):

    """
    Function to get R2

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    R2_array: An array of R2s for each output
    """

    nom_list=[]
    denom_list=[]
    R2_list=[] 
    for i in range(y_test.shape[1]): #Loop through outputs
        y_mean=np.mean(y_test[:,i])
        nom = np.sum((y_test_pred[:,i]-y_test[:,i])**2)
        denom = np.sum((y_test[:,i]-y_mean)**2)
        nom_list.append(nom)
        denom_list.append(denom)
        R2_list = [nom_list, denom_list]
        R2_array=np.array(R2_list)
        
        #R2_list = [nom, denom] #Append R2 nom, denom to the list
        #R2_list = [len(y_test), len(y_test_pred)]
        #R2_list = [y_test, y_test_pred]
        #R2_array=R2_list
        #R2_array=np.array(R2_list)
    return R2_array 


########## R-squared (R2) for XY combined ##########

def compute_XY_FVAF(x_nom,x_denom,y_nom,y_denom):
    nom = x_nom + y_nom
    denom = x_denom + y_denom 
    XY_FVAF = 1 - (nom/denom)
    return XY_FVAF


########## Pearson's correlation (rho) ##########

def get_rho(y_test,y_test_pred):

    """
    Function to get Pearson's correlation (rho)

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    rho_array: An array of rho's for each output
    """

    rho_list=[] #Initialize a list that will contain the rhos for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute rho for each output
        y_mean=np.mean(y_test[:,i])
        rho=np.corrcoef(y_test[:,i].T,y_test_pred[:,i].T)[0,1]
        rho_list.append(rho) #Append rho of this output to the list
    rho_array=np.array(rho_list)
    return rho_array #Return the array of rhos
