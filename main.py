import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def app():
    apple_training_complete = pd.read_csv('dataset/AAPL_2017.csv')

    #cabeçalho dos dados
    # Date  Open    High    Low    Close   Adj Close   Volume
    # data; valor abertura; valor máximo; valor mínima; valor de fechamento; asd;  volume de negociações

    #removendo a coluna do preço de fechamento
    apple_training_processed = apple_training_complete.iloc[:, 1:2].values
    
    #normalizando os dados
    scaler = MinMaxScaler(feature_range = (0, 1))
    apple_training_scaled = scaler.fit_transform(apple_training_processed)

    #print(len(apple_training_scaled))
    #return
    
    #alterando o formato dos dados para comportar a LSTM
    features_set = []
    labels = []
    
    for i in range(60, 1259):
        features_set.append(apple_training_scaled[i-60:i, 0])
        j = apple_training_scaled[i, 0]
        labels.append(j)

    features_set, labels = np.array(features_set), np.array(labels)


    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))


    checkpoint_filepath = 'checkpoints/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=False,
    verbose=1)   
    
    features_set_shape_1 = features_set.shape[1]
    model = buildModel(features_set_shape_1)

    model.load_weights(checkpoint_filepath)

    model.fit(features_set, labels, epochs = 3, batch_size = 32)


    apple_testing_complete = pd.read_csv('dataset/AAPL_jan_2018.csv')
    apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values

    apple_total = pd.concat((apple_training_complete['Open'], apple_testing_complete['Open']), axis=0)

    test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values


    test_inputs = test_inputs.reshape(-1,1)
    test_inputs = scaler.transform(test_inputs)

    test_features = []
    for i in range(60, 80):
        test_features.append(test_inputs[i-60:i, 0])

    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

    predictions = model.predict(test_features)

    predictions = scaler.inverse_transform(predictions)


    plt.figure(figsize=(10,6))
    plt.plot(apple_testing_processed, color='blue', label='Actual Apple Stock Price')
    plt.plot(predictions , color='red', label='Predicted Apple Stock Price')
    plt.title('Apple Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Apple Stock Price')
    plt.legend()
    plt.show()

def buildModel(features_set_shape_1):
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set_shape_1, 1)))

    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model

app()