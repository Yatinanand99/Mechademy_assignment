
import pandas as pd
import pickle


def __rf_reg_training__(dataset,n_estimator=10):
    dataset = dataset.fillna(method='ffill')
    X = dataset.iloc[:,[1,2,3,4,5,7]].values
    y = dataset.iloc[:,6].values
    from sklearn.preprocessing import MinMaxScaler
    Sc_X = MinMaxScaler()
    X = Sc_X.fit_transform(X)
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=n_estimator,random_state=0)
    regressor.fit(X, y)
    filename = 'finalized_model.sav'
    pickle.dump(regressor, open(filename, 'wb'))

def __nn_reg_train__(dataset, epochs = 40, batch_size = 512):
    dataset = dataset.fillna(method='ffill')
    X = dataset.iloc[:,[1,2,3,4,5,7]].values
    y = dataset.iloc[:,6].values
    from sklearn.preprocessing import MinMaxScaler
    Sc_X = MinMaxScaler()
    X = Sc_X.fit_transform(X)
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.models import Sequential
    from keras.callbacks import ModelCheckpoint,TensorBoard
    
    regressor = Sequential()
    
    regressor.add(Dense(units = 512, kernel_initializer='uniform', input_dim = X.shape[1]))
    
    regressor.add(Dense(units = 1024, kernel_initializer='uniform'))
    regressor.add(Dense(units = 512, kernel_initializer='uniform'))
    regressor.add(Dense(units = 512, kernel_initializer='uniform'))
    regressor.add(Dropout(0.25))
    
    regressor.add(Dense(units = 512, kernel_initializer='uniform'))
    regressor.add(Dense(units = 1024, kernel_initializer='uniform'))
    regressor.add(Dense(units = 1024, kernel_initializer='uniform'))
    regressor.add(Dropout(0.25))
    
    regressor.add(Dense(units = 1024, kernel_initializer='uniform'))
    regressor.add(Dense(units = 512, kernel_initializer='uniform'))
    regressor.add(Dense(units = 1024, kernel_initializer='uniform'))
    regressor.add(Dropout(0.25))
    
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model_json = regressor.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    checkpoints = []
    checkpoints.append(ModelCheckpoint('checkpoints/weights_alpha_.{epoch:02d}-{loss:.2f}.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir='tensorboard_logs/logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

    regressor.fit(X, y, epochs = epochs, batch_size = batch_size, callbacks = checkpoints)


def __rf_reg_testing__(dataset,save_file_name):
    dataset = dataset.fillna(method='ffill')
    X = dataset.iloc[:,[1,2,3,4,5,7]].values
    y = dataset.iloc[:,6].values
    from sklearn.preprocessing import MinMaxScaler
    Sc_X = MinMaxScaler()
    X = Sc_X.fit_transform(X)
    loaded_model = pickle.load(open(save_file_name, 'rb'))
    result = loaded_model.score(X, y)
    return result

def __nn_regressor_prediction__(dataset,weight_loc):
    dataset = dataset.fillna(method='ffill')
    X = dataset.iloc[:,[1,2,3,4,5,7]].values
    y = dataset.iloc[:,6].values
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    regressor = model_from_json(loaded_model_json)
    regressor.load_weights(weight_loc)
    y_pred = regressor.predict(X)
    y_test = y
    return y_pred,y_test



#----------------------------------------------------------------------------------#
#Call this for training
#Training on Random Forest with n_estimator to be given
dataset = pd.read_csv("Expander_data.csv")
__rf_reg_training__(dataset,n_estimator=10)
#----------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------#
#Call this for training
#Training on a neural network where epochs and batch_size to be given accordingly
dataset = pd.read_csv("Expander_data.csv")
__nn_reg_train__(dataset,epochs=100,batch_size=512)
#----------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------#
#Call this function for testing on a dataset in the format as in training
#This function tests the result on Random Forest model
#Saved file name is to be given
#This function returns the accuracy of the model

dataset = pd.read_csv("abc.csv") ## Load dataset by replacing abc.csv with proper testing dateset
model_accuracy = __rf_reg_testing__(dataset,'finalized_model.sav')
#----------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------#
#Call this function for testing on a dataset in the format as in training
#This function tests the result on a Neural Network
#Saved weights file location and name is to be given
#This function returns predicted values and actual values

dataset = pd.read_csv("abc.csv") ## Load dataset by replacing abc.csv with proper testing dateset
predicted_values,actual_values = __nn_regressor_prediction__(dataset,'checkpoints/weights_alpha_.03-28.82.h5')
#----------------------------------------------------------------------------------#