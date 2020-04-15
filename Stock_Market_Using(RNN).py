import numpy as np
from keras.layers import Dense,LSTM
from keras.layers import Dropout
import pandas as pd
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

# configure graphing visuals/size
sns.set_style("darkgrid")
plt.figure(figsize=(12, 5))

#Second Dataset TATA-Dataset
#Training DataSet
url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'

#Test Dataset

url_1 = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
#Using Pandas to read the data file


#price_train = pd.read_csv('/home/varun/Desktop/SP500_train.csv')
price_train=pd.read_csv(url)
price_test=pd.read_csv(url_1)

#price_test = pd.read_csv('/home/varun/Desktop/SP500_test.csv')

""" 
    Creates list of values from stock csv - possible columns with indexes:
    Date(0), Open(1), High(2), Low(3), Close(4), Adj Close(5), Volume(6)
    Here, I have taken adj_close price as the target parameter
"""

trainingset = price_train.iloc[:,1:2].values
test_set = price_test.iloc[:,1:2].values

"Here the raw data is Normalized using Min_MAx Trasnformation for the model"
min_max_scaler = MinMaxScaler(feature_range=(0,1))
scaled_training_set = min_max_scaler.fit_transform(trainingset)

"""
    Breaks the values into the data and labels with the time window
    (window) days of data, next day as the label
"""
X_train=[]
Y_train=[]

for i in range(40,1258):
    
    X_train.append(scaled_training_set[i-40:i,0])
    
    Y_train.append(scaled_training_set[i,0])
    
X_train= np.array(X_train)

Y_train=np.array(Y_train)

X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))   

"LSTM Model is being created"

model=Sequential()

model.add(LSTM(units=200, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.4))
model.add(LSTM(units=80, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80))
model.add(Dropout(0.2))
model.add(Dense(units=1))    

model.compile(optimizer='adam',
              loss='mean_squared_error')

model.fit(X_train,Y_train,epochs=100,batch_size=40)

#Here model has been saved to be used for the future process
model.save('model_stock.h5')

"Here the test data is being again breaked and noramlized like the training data"
#Here change the name of the column according to the parameter selected above and the name should same as in the dataset i.e csv file
dataset_total=pd.concat((price_train['Open'],price_test['Open']),axis=0)

final_dataset=dataset_total[len(dataset_total)-len(price_test)-40:].values

final_dataset=final_dataset.reshape(-1,1)

final_dataset=min_max_scaler.fit_transform(final_dataset)

X_test=[]

for i in range(40,len(price_test)+40):
    X_test.append(final_dataset[i-40:i,0])
    
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

#Predicitons are being made using the model on the test data
predictions = model.predict(X_test)

#Here the predcitions are Denormalized usin inverse TRansform function to give the appropriate value 
predictions=min_max_scaler.inverse_transform(predictions)

print(predictions)

#Graph is being plotted for the Test Data vs The Predicited data by the model
#plt.plot(test_set,color='green', label='Actual Price of S&P 500')
plt.plot(test_set,color='green', label='Actual Price of TATA')
plt.plot(predictions, color='red',label='Predicited Price of TATA')
#plt.plot(predictions, color='red',label='Predicited Price of S&P 500')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

df = DataFrame(test_set)

sns.lineplot(data=df, dashes=False)
plt.show()

df_1=DataFrame(predictions)
sns.lineplot(data=df_1, dashes=False)
plt.show()