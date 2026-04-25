import copy
import os

import keras.callbacks

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import regularizers

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


np.set_printoptions(precision=3, suppress=True)

def Normalise(x):
    x_min = x.min()
    x_max = x.max()
    for i in range(len(x)):
        x[i] = (x[i] - x_min) / (x_max - x_min)
    return x

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show()

def train(TRAINING,systems):
    result_dict = {}
    for current_system in systems:
        datasets_location = 'datasets/{}'.format(current_system)
        csv_files = [f for f in os.listdir(datasets_location) if f.endswith('.csv')]
        if len(csv_files) == 0:
            print("No csv files in {}".format(datasets_location))
        for csv_file in csv_files:
            EPOCHS = 1000
            BATCH_SIZE = 100
            VALIDATION_SPLIT = 0.2
            TRAINING = True
            filename = csv_file.strip('.')
            print('\n> System: {}, Dataset: {} Training data fraction: 0.8'.format(current_system,csv_file))
            df = pd.read_csv(os.path.join(datasets_location, csv_file))
            X = df.iloc[:,:-1].values
            y = df.iloc[:,-1].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

            for i in range(X_train.shape[1]):
                X_train[:,i] = Normalise(X_train[:,i])
            for i in range(X_test.shape[1]):
                X_test[:,i] = Normalise(X_test[:,i])
            val_loss = -1
            lowest_val_loss = np.inf
            extra_hidden_layers = 0
            if os.path.exists('models/{}_{}.keras'.format(current_system, filename)):
                best_model = load_model('models/{}_{}.keras'.format(current_system, filename))
            else:
                #HYPERPARAMETER TUNING
                if TRAINING:
                    while val_loss <= lowest_val_loss:
                        model = Sequential()
                        model.add(Dense(units=X_train.shape[1], kernel_regularizer=regularizers.l1(0.01) ,input_dim=X_train.shape[1], activation='relu'))
                        model.add(Dense(units=X_train.shape[1], activation='relu'))
                        for i in range(extra_hidden_layers):
                            model.add(Dense(units=X_train.shape[1], activation='relu'))

                        model.add(Dense(units=1))
                        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=13, restore_best_weights=True)
                        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae', 'mape'])
                        history = model.fit(X_train, y_train, epochs=15, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, callbacks=[early_stop])
                        hist = pd.DataFrame(history.history)
                        val_loss = min(history.history['val_loss'])
                        if val_loss < lowest_val_loss:
                            lowest_val_loss = val_loss
                            best_model = copy.deepcopy(model)
                        else:
                            print("Overfitted, ending loop")
                        extra_hidden_layers = extra_hidden_layers + 1
                    #WEIGHT TUNING
                    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                    history = best_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,validation_split=VALIDATION_SPLIT, callbacks=[early_stop])
            try:
                y_pred = best_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                print("MAE: {}".format(mae))
                print("MAPE: {}".format(mape))
                result_dict["{}_{}".format(current_system,filename)] = [mae, mape]
                best_model.save('models/{}_{}.keras'.format(current_system,filename))
            except NameError:
                print("No model found for {}, {}".format(current_system, filename))

    f = open('neural_network_results.txt','w')
    f.write(json.dumps(result_dict))
    f.close()

def main():
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    #systems = ['jump3r']
    loop = True
    while loop:
        print("Deep Neural Network Tool")
        print("1. Start training")
        print("2. Start testing")
        print("3. Change systems")
        print("4. Exit")
        try:
            choice = int(input("Enter the number: "))
            if choice == 1:
                train(True, systems)
            elif choice == 2:
                train(False, systems)
            elif choice == 3:
                print("The current systems are: {}".format(systems))
                print("1. Add system")
                print("2. Remove system")
                try:
                    choice2 = int(input("Enter the number: "))
                    if choice2 == 1:
                        sys = input("Enter the name of the system to add: ")
                        if sys not in systems:
                            systems.append(sys)
                        else:
                            print("System already added")
                    elif choice2 == 2:
                        sys = input("Enter the name of the system to remove: ")
                        if sys not in systems:
                            print("System does not exist")
                        else:
                            systems.remove(sys)
                            print("System removed")
                except ValueError:
                    print("Invalid input, enter a number")
            elif choice == 4:
                loop = False
            else:
                print("Invalid number")
        except ValueError:
            print("Invalid input, enter a number")


main()