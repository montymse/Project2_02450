# Script from exercise 8.2.6 & 7.2.1 reused to solve Question B.2 in project 2
import numpy as np
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net
import sklearn.linear_model as lm
from scipy import stats
import scipy.stats as st
import pandas as pd
from toolbox_02450 import rlr_validate



# Import data
data = pd.read_csv('Glass data/glass.data',header=None)

# Get tags
header = ["RI","NA","MG","AL","SI","K","CA","BA","FE","Type"]

# Drop first column and add tags
data = data.drop(columns=0)
data.columns=header

# Prepare and clean data
X = data.drop(columns='Type')
attributeNames = X.columns.tolist()[1:]
X=X.to_numpy()

y  = X[:,0]
X = X[:,1:]

N, M = X.shape
C = 2

# Normalize data
X = stats.zscore(X)
                
## Normalize and compute PCA (change to True to experiment with PCA preprocessing)
do_pca_preprocessing = True
if do_pca_preprocessing:
    Y = stats.zscore(X,0)
    U,S,V = np.linalg.svd(Y,full_matrices=False)
    V = V.T
    #Components to be included as features
    k_pca = 3
    X = X @ V[:,:k_pca]
    N, M = X.shape

lambdas = np.power(10.,range(-5,8))

# Parameters for neural network classifier
n_hidden_units = 2      # number of hidden units
n_replicates = 2        # number of networks trained in each k-fold
max_iter = 10000

# K-fold crossvalidation
K = 5
K2 = 5          
CV = model_selection.KFold(K, shuffle=True)
CV2 = model_selection.KFold(K2, shuffle=True)

# Loss list
MSE_Loss = np.empty((K,4))


# Model for ANN
model_ANN = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )


loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

errors = [] # make a list for storing generalizaition error in each loop

mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

for (c, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nOuter Crossvalidation fold: {0}/{1}'.format(c+1,K))
    X_train0 = X[train_index]
    y_train0 = y[train_index]
    X_test0 = X[test_index]
    y_test0 = y[test_index]

    for (k, (train_index, test_index)) in enumerate(CV.split(X_train0,y_train0)): 
        print('\n\tInner Crossvalidation fold: {0}/{1}'.format(k+1,K))
        
    #%% Linear regression    
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        internal_cross_validation = 10
          
        X0 = (X_train - np.mean(X_train,0) ) / np.std(X_train,0)
        X0 = np.concatenate((np.ones((X_train.shape[0],1)),X_train),1)
    
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X0, y_train, lambdas, internal_cross_validation)
    
        model_LR = lm.LinearRegression().fit(X_train, y_train)
    
        MSE_Loss[k,0] = np.square(y_test-model_LR.predict(X_test)).sum()/y_test.shape[0]
    
    #%% Baseline    
        MSE_Loss[k,1] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
        
    #%% ANN          
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index].reshape(len(y[train_index]),1))
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index].reshape(len(y[test_index]),1))
        
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model_ANN,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
    
        # Determine estimated class labels for test set
        y_test_est = net(X_test)
        
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        MSE_Loss[k,2] = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    
    
    print('\nOuter Crossvalidation fold: {0}/{1}'.format(c+1,K))
    print('\n Lambda: {0}'.format(opt_lambda))
    print('\n Loss Linear Regression: {0}'.format(MSE_Loss[k,0]))
    print('\n Loss Baseline: {0}'.format(MSE_Loss[k,1]))
    print('\n Loss ANN: {0}'.format(MSE_Loss[k,2]))
    print('\n----------------------------------------------------------------------\n')



   

def compare(zA, zB, alpha = 0.05):
    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    z = zA - zB
    CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

    print('\nConfidence Interval: {0}'.format(CI))
    print('P-value:               {0}\n'.format(p))
    

print('\n\n-----------------ANN vs Linear Regression------------------------------')
compare(MSE_Loss[:,2], MSE_Loss[:,0])

print('-----------------ANN vs Baseline---------------------------------------')
compare(MSE_Loss[:,2], MSE_Loss[:,1])

print('-----------------Linear Regression vs Baseline-------------------------')
compare(MSE_Loss[:,0], MSE_Loss[:,1])