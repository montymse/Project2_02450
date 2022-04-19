import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os 

# Set directory to same as the script
os.chdir(os.path.dirname(__file__))

# Import data
data = pd.read_csv('Glass data/glass.data',header=None)

# Get tags
header = ["RI","NA","MG","AL","SI","K","CA","BA","FE","Type"]

# Drop first column and add tags
data = data.drop(columns=0)
data.columns=header

# Prepare and clean data
X = data.drop(columns='Type') 
X=X.to_numpy()


y  = X[:,0]
X = X[:,1:]


# Descriptive Statistics
df_describe = pd.DataFrame(X)
print (df_describe.describe())


#Standadize (Part A.1 in project description)

m = X.mean(axis=0)
sd = np.std(X,0)
X_std = (X-m)/sd