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
import glob

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
df.rename(columns={' Label':'y'},inplace=True)
print((df.columns))
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df.y=le.fit_transform(df.y)
print(le.inverse_transform(list(i for i in range (0,max(df.y)+1))))

remv=np.array([0,1,7,8,9,11,12,13,17,19,20,21,22,24,25,26,27,29,30,39,40,41,43,71,73,74,75,77,32,34,57,58,59,60,61,62])
df.drop(df.columns[remv], axis = 1, inplace = True)
print(df.shape)
##############################################################################################################
df_test = df[df.y != 0]
X = df_test.loc[:, df_test.columns != 'y']
y=df_test.loc[:, df_test.columns == 'y']
print(X.shape, y.shape)
##############################################################################################################
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
Y=enc.fit_transform(y)
print(Y.shape)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_final=scaler.fit_transform(X)
##############################################################################################################
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_final=scaler.fit_transform(X)
x_train, x_test,y_train, y_test = train_test_split(X_final,Y,test_size=0.15)
##############################################################################################################
model=Sequential()
model.add(Dense(units=64,input_shape=(42,), activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=14, activation='softmax'))
model.summary()
##############################################################################################################
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history= model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=1024)
model.save('attacks.h5')