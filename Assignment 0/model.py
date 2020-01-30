
import pandas as pd
import numpy as np
dataset = pd.read_csv("Weather_data.csv")

def __train__(dataset, days = 3, epochs = 50, batch_size = 1024):
    ndataset = dataset.fillna(method='ffill')
    X = ndataset.iloc[:,[2,3,4,6,8,9,10,11,12,13,14,19]].values
    y = ndataset.iloc[:,11:12].values
    j = 0
    for i in X[:,4]:
        if i == -9999:
            X[j,4] = X[j-1,4]
        j+=1
        
    #Scaling the data
    from sklearn.preprocessing import MinMaxScaler
    Sc_X = MinMaxScaler()
    X = Sc_X.fit_transform(X)
    
    #Creating the training data, taking 3 days to predict for 1 day
    X_train = []
    y_train = []
    previous_days = 24 * days
    for i in range(previous_days,len(X)):
        X_train.append(X[i-previous_days:i, :])
        y_train.append(X[i, :])
    X_train, y_train = np.array(X_train), np.array(y_train)

    #Making the Prediction model
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    
    # Initialising the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.2))
    
    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 100, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 100, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 100))
    regressor.add(Dropout(0.2))
    
    # Adding the output layer
    regressor.add(Dense(units = 12))
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    #Saving the model in JSON format
    model_json = regressor.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    
    from keras.callbacks import TensorBoard
    checkpoints=[]
    checkpoints.append(TensorBoard(log_dir='tensorboard_logs/logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))
    
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,callbacks = checkpoints)
    
    regressor.save('model_weight.h5')

def __test__(dataset, date_time, weight_loc, days = 3):
    ndataset = dataset.fillna(method='ffill')
    X = ndataset.iloc[:,[2,3,4,6,8,9,10,11,12,13,14,19]].values
    y = ndataset.iloc[:,11:12].values
    j = 0
    for i in X[:,4]:
        if i == -9999:
            X[j,4] = X[j-1,4]
        j+=1
        
    #Scaling the data
    from sklearn.preprocessing import MinMaxScaler
    Sc_X = MinMaxScaler()
    X = Sc_X.fit_transform(X)
    
    #Loading the regressor
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    regressor = model_from_json(loaded_model_json)
    regressor.load_weights(weight_loc)
    
    loc = ndataset.loc[ndataset['datetime_utc']==date_time].index[0]
    previous_days = 24 * days
    #Making the test set
    X_test = []
    y_test = []
    y_actual = []
    X_test.append(X[loc - previous_days:loc, :])
    X_test = np.array(X_test)
    X_test.reshape(-1,X_test.shape[0],X_test.shape[1])
    for i in range(0, 24):
        y_pred = regressor.predict(X_test)
        X_test = np.append(X_test,np.atleast_3d(y_pred).reshape(1,1,y_pred.shape[1]),axis=1)
        X_test = X_test[:,-previous_days:,:]
        y_test.append(y_pred)
        y_actual.append(y[loc+i, 0])
    y_test, y_actual = np.array(y_test), np.array(y_actual)
    
    y_predict = []
    for i in range(len(y_test)):
      y_pred = Sc_X.inverse_transform(y_test[i])
      y_predict.append(y_pred[:,7])
    y_test = y_actual
    
    #Plotting the results
    from matplotlib import pyplot as plt
    plt.title("Prediction vs Actual temp")
    plt.plot(y_predict)
    plt.plot(y_test)
    plt.ylabel('Temperature')
    plt.xlabel('Dates')
    plt.legend(['Prediction', 'Actual'], loc='upper left')
    plt.show()



#----------------------------------------------------------------------------------------#
#Call this function for training on the dataset
__train__(dataset, days = 3, epochs = 100,batch_size = 1024)
#----------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------#
#Call this function for testing on the dataset
#Provide it the dataset
#Give the date on which you need to predict(Ensure that it must be atleast number of days ahead you are selecting)
#Example if you are selecting days be 3 then date to be predicted must be atleast 3 days ahead from starting
#Give it the location of weight file
__test__(dataset, '19970207-10:00','model_weight.h5',days = 3)
#----------------------------------------------------------------------------------------#

