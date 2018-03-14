from keras.models import Sequential
from keras.layers import Dense
import numpy
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
numpy.random.seed(7)

dataframe = read_csv('1.csv', usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
xscaler = MinMaxScaler()
yscaler = MinMaxScaler()
X = dataset[:,0:17]
#print(X)
X_scale = xscaler.fit_transform(X)
print(X)
Y = dataset[:,18]
#print(Y)
Y = Y.reshape(-1, 1)
Y = yscaler.fit_transform(Y)
#print(Y)
mind = xscaler.data_min_
maxd = xscaler.data_max_
ranged = xscaler.data_range_
print(mind)
print(maxd)
print(ranged)
model = Sequential()
model.add(Dense(12, input_dim=17, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

model.fit(X_scale, Y, epochs=100, batch_size=10, shuffle=True)

scores = model.evaluate(X_scale, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




trainPredict = model.predict(X_scale)
#print(trainPredict)




real_prediction = yscaler.inverse_transform(trainPredict)
#print(real_prediction)

dataframe_predict = read_csv('2.csv', usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], engine='python')
dataset_predict = dataframe_predict.values
dataset_predict = dataset_predict.astype('float32')
D = dataset_predict[:,0:17]
print(D)
print(type(D))
z = D.shape
print(type(z))
i=0
while i != int(z[1]):
    D[0,i] = (D[0,i] - mind[i]) / ranged[i]
    i = i + 1
print(D)
#D = xscaler.fit_transform(D)
#print(D)
DPredict = model.predict(D)
print(DPredict)
yd = yscaler.inverse_transform(DPredict)
print(yd)
