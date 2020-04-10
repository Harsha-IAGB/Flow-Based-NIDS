import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve
from keras.models import load_model
from sklearn import preprocessing
import glob

#df = pd.read_csv("51.csv") 
#print(df.shape)

all_files = glob.glob("*.csv")
print(all_files)
df = []
for filename in all_files:
    x = pd.read_csv(filename, index_col=None, header=0)
    df.append(x)
df = pd.concat(df, axis=0, ignore_index=True)

print(df.shape)
print(list(df.dtypes).count('float64'),list(df.dtypes).count('int64'),list(df.dtypes).count('O'))

a=np.array(df.iloc[:,14])
b=np.array(df.iloc[:,15])
r=[]
for i in range(0,df.shape[0]):
    a[i]=float(a[i])
    b[i]=float(b[i])
    if(a[i]==float('inf') or b[i]==float('inf')):
        r.append(i)
print(len(r))
df.iloc[:,14]=a
df.iloc[:,15]=b

indexes_to_keep = set(range(df.shape[0])) - set(r)
df = df.take(list(indexes_to_keep))
print(df.shape)
df['Flow Bytes/s']=pd.to_numeric(df['Flow Bytes/s'],errors='coerce')
df[' Flow Packets/s']=pd.to_numeric(df[' Flow Packets/s'],errors='coerce')
df=df.dropna()
print(df.shape)
print(list(df.dtypes).count('float64'),list(df.dtypes).count('int64'),list(df.dtypes).count('O'))
#print(df.dtypes)
df.rename(columns={' Label':'y'},inplace=True)
print((df.columns))
le = preprocessing.LabelEncoder()
df.y=le.fit_transform(df.y)
print(le.inverse_transform(list(i for i in range (0,max(df.y)+1))))

remv=np.array([0,1,7,8,9,11,12,13,17,19,20,21,22,24,25,26,27,29,30,39,40,41,43,71,73,74,75,77,32,34,57,58,59,60,61,62])
df.drop(df.columns[remv], axis = 1, inplace = True)
print(df.shape)
############################################################################################################

LOOKBACK = 20
benign_data=df[df.y==0]
benign_data = benign_data.loc[:, benign_data.columns != 'y']
def shift(df,shift_by):
    shift_by = abs(shift_by)
    df.y = df.y.shift(-shift_by)
    df = df[:-shift_by]
    df.y = df.y.astype('int32')
    return df
def AddTemporalDimension(df,lookback):
    new_df = np.empty((df.shape[0],lookback,df.shape[1]))
    for ii in range(lookback,df.shape[0]):
        new_df[ii] = df[ii-lookback:ii]
    return new_df
def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)
def scale(X, scaler):
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])
    return X

df_1=AddTemporalDimension(X[0:100],LOOKBACK)
print("Test_1:",df_1.shape)
scaler = MinMaxScaler().fit(flatten(df_1))
x1_scaled_shaped = scale(df_1, scaler)
print(x1_scaled_shaped.shape)
test_x1_predictions = lstm_ae.predict(x1_scaled_shaped)
mse_1 = np.mean(np.power(flatten(x1_scaled_shaped) - flatten(test_x1_predictions), 2), axis=1)


df_2=AddTemporalDimension(benign_data[0:100],LOOKBACK)
print("Test_1:",df_2.shape)
scaler = MinMaxScaler().fit(flatten(df_2))
x2_scaled_shaped = scale(df_2, scaler)
print(x2_scaled_shaped.shape)
test_x2_predictions = lstm_ae.predict(x2_scaled_shaped)
mse_2 = np.mean(np.power(flatten(x2_scaled_shaped) - flatten(test_x2_predictions), 2), axis=1)
##############################################################################################################
lstm_ae=load_model('output\lstm_ae.h5')
attacks=load_model('attacks.h5')
##############################################################################################################
print(test_x1_predictions.shape)
print(mse_1.shape)
print(test_x2_predictions.shape)
print(mse_2.shape)
##############################################################################################################
ben=[]
for i in range(mse_2.shape[0]):
    if mse_2[i]>0.10:
        ben.append(np.argmax(attacks.predict(test_x2_predictions[i,19,:].reshape(1,42))))
    else:
        ben.append('benign')
atk=[]
for i in range(mse_1.shape[0]):
    if mse_1[i]>0.10:
        atk.append(np.argmax(attacks.predict(test_x1_predictions[i,19,:].reshape(1,42))))
    else:
        atk.append('benign')
##############################################################################################################
print(ben,'\n')
print("No. of correctly classified benign packets out of 100:",ben.count('benign'),'\n')
print(atk,'\n')
print("No. of correctly classified attack packets out of 100:",100-atk.count('benign'),'\n')