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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve
import glob
from sklearn import preprocessing


DATA_SPLIT_PCT = 0.1
LOOKBACK = 20
BATCH_SIZE = 512
EPOCHS = 100
TRAIN_SIZE=200000
LIMIT=12

all_files = glob.glob("MachineLearningCSV/MachineLearningCVE/*.csv")
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
df_raw = df.copy()
print("Shifting.....")
def shift(df,shift_by):
    shift_by = abs(shift_by)
    df.y = df.y.shift(-shift_by)
    df = df[:-shift_by]
    df.y = df.y.astype('int32')
    return df

df = df_raw.copy()
shift_by = -2
df = shift(df,shift_by)
assert len(df_raw)-abs(shift_by) == len(df)
assert len(df[df.y.isna()]) == 0

df_train = df[df.y == 0]
df_train = df_train.loc[:, df_train.columns != 'y']
df_test = df[df.y != 0]
df_test = df_test.loc[:, df_test.columns != 'y']
df=df_train

print("Temporizing...")
def AddTemporalDimension(df,lookback):
    new_df = np.empty((df.shape[0],lookback,df.shape[1]))
    for ii in range(lookback,df.shape[0]):
        new_df[ii] = df[ii-lookback:ii]
    return new_df
df=AddTemporalDimension(df,LOOKBACK)
print(df.shape)
############################################################################################################
timesteps=LOOKBACK
n_features=df.shape[2]
lstm_ae = Sequential()
lstm_ae.add(LSTM(32, input_shape=(timesteps, n_features), activation = 'relu', return_sequences = True))
lstm_ae.add(LSTM(16, activation = 'relu',return_sequences = False))
lstm_ae.add(RepeatVector(timesteps))
lstm_ae.add(LSTM(16, activation = 'relu',return_sequences = True))
lstm_ae.add(LSTM(32, activation = 'relu',return_sequences = True))
lstm_ae.add(TimeDistributed(Dense(n_features)))
lstm_ae.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
lstm_ae.summary()
############################################################################################################
x_test_final=[]
history=[]
for epoch in range(0,LIMIT):
    print("ITERATION:",epoch)
    lets_take=df[(epoch)*TRAIN_SIZE:min(((epoch)+1)*TRAIN_SIZE,df.shape[0])]
    print(lets_take.shape)
    x_train, x_test= train_test_split(np.array(lets_take),test_size=DATA_SPLIT_PCT)
    x_test_final.append(x_test)
    print(f"x_train:{x_train.shape} x_test:{x_test.shape} ")
    n_samples = x_train.shape[0] 
    n_features = x_train.shape[1] 
    
    def flatten(X):
        flattened_X = np.empty((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1]-1), :]
        return(flattened_X)

    def scale(X, scaler):
        for i in range(X.shape[0]):
            X[i, :, :] = scaler.transform(X[i, :, :])
        return X
    scaler = MinMaxScaler().fit(flatten(x_train))
    x_train_scaled_shaped = scale(x_train, scaler)
    timesteps =  x_train_scaled_shaped.shape[1] # equal to the lookback
    n_features =  x_train_scaled_shaped.shape[2] # equal to features

    lstm_autoencoder_history=lstm_ae.fit(x_train_scaled_shaped,x_train_scaled_shaped,epochs=EPOCHS,verbose=1,batch_size=BATCH_SIZE,
                                         validation_split=DATA_SPLIT_PCT).history
    history.append(lstm_autoencoder_history)
############################################################################################################
df_1=AddTemporalDimension(df_test,LOOKBACK)
print("Test_1:",df_1.shape)
scaler = MinMaxScaler().fit(flatten(df_1))
x1_scaled_shaped = scale(df_1, scaler)
print(x1_scaled_shaped.shape)
timesteps =  x1_scaled_shaped.shape[1] # equal to the lookback
n_features =  x1_scaled_shaped.shape[2] # equal to features
print(f"timesteps: {timesteps}  n_features:{n_features}")
test_x1_predictions = lstm_ae.predict(x1_scaled_shaped)
mse_1 = np.mean(np.power(flatten(x1_scaled_shaped) - flatten(test_x1_predictions), 2), axis=1)

mse_0=np.empty((20000,))
for x_test in x_test_final:
    print("Test:",x_test.shape)
    scaler = MinMaxScaler().fit(flatten(x_test))
    x_test_scaled_shaped = scale(x_test, scaler)
    print(x_test_scaled_shaped.shape)
    timesteps =  x_test_scaled_shaped.shape[1] # lookback
    n_features =  x_test_scaled_shaped.shape[2] # features
    print(f"timesteps: {timesteps}  n_features:{n_features}")

    test_x_predictions = lstm_ae.predict(x_test_scaled_shaped)
    mse_temp=np.mean(np.power(flatten(x_test_scaled_shaped) - flatten(test_x_predictions), 2), axis=1)
#     print(mse_temp.shape)
    print(np.mean(mse_temp),np.mean(mse_1))
    mse_0 = np.append(mse_0, mse_temp)

print("Mean_0:",np.mean(mse_0),"  Mean_1:",np.mean(mse_1))
print("Min/Max")
print("0--",np.min(mse_0),np.max(mse_0))
print("1--",np.min(mse_1),np.max(mse_1))
############################################################################################################
print("shape of mse_0", mse_0.shape)
############################################################################################################
lstm_ae.save('lstm_ae.h5')