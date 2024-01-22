import pandas as pd 
import numpy as np

import xgboost as xgb
import matplotlib.pyplot as plt
df = pd.read_csv('airfoil_self_noise.dat',  sep='\t', header = None)
df.columns = ['Frequency','Angle of attack',' Chord length','Free-stream velocity','Suction side','Scaled sound pressure level']
print(df.isnull().sum())
print(df)
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1]
Y = df.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=2)
import seaborn as sns 
sns.pairplot(X_train)
plt.show()
regress = xgb.XGBRegressor()
regress.fit(X_train,Y_train)
import pickle
pickle.dump(regress,open('model.pkl','wb'))
