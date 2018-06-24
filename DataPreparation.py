import csv
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as skp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np










print("Data Preparation")
print()
csv_file = open("BrentDataset.csv")
reader = csv.reader(csv_file)
#print(next(reader))

dta_v = pd.read_csv("BrentDataset.csv")

def dta_vanilla ():
    return dta_v


def dta_preparation():
    dta = dta_v.reset_index()   #setzt Index zurück
    #print(dta.head())
    date_split = dta["index"].str.split(expand=True)    #neuer Datensatz aus Index, gesplittet in Monat und Tag
    del dta["index"]                                    #löscht Spalte Index aus ursprünglichem Datensatz
    dta = pd.concat((date_split, dta), axis=1)          #Verbindet beide Datensätze
    dta = dta.rename(columns={0: "Month", 1: "Day", "Date": "Year", "Price": "Close", "Vol.": "Volume", "Change %": "Change"})  #Umbennenungen
    dta.Change = [x.strip("%") for x in dta.Change]     #löscht %-Zeichen aus Spalte Change für jeden Datensatz
    dta.Volume = [x.replace("-","0") for x in dta.Volume]   #ersetzt "-"-Zeichen mit "0"
    dta[["Volume"]] = dta[["Volume"]].astype(float)         #Typanpassung
    dta[["Change"]] = dta[["Change"]].astype(float)         #Typanpassung
    dta[["Year"]] = dta[["Year"]].astype(str)               #Typanpassung
    dta["Date"] = dta["Year"] + "/" + dta["Month"] + "/" + dta["Day"]       #neuer Datensatz mit Spalte Date aus Year, Month und Day
    dta_date = pd.to_datetime(dta["Date"])              #Typanpassung für Date
    del dta["Date"]                             #löscht alte Spalte "Date"
    dta_final = pd.concat((dta, dta_date), axis=1)      #verbindet alten und neuen Datensatz
    dta_final.sort_values(by=["Date"], ascending=True, inplace=True)    #Sortierung der Daten nach Date
    #print(list(dta_final.columns.values))
    dta_final = dta_final[["Date","Year","Month","Day","Low","High","Open","Close","Change","Volume"]]  #finale Anordnung der Spalten
    #print(dta_final.head())
    return dta_final

def dta_rescaled(dta_final):
    #x = dta_final.values[:, 5:8]
    #y = dta_final[:, 7]
    #x = skp.MinMaxScaler(feature_range=(0, 1)).fit_transform(x)
    scaler =skp.MinMaxScaler()
    dta_final[["Low", "High", "Open", "Close", "Change", "Volume"]] = scaler.fit_transform(dta_final[["Low", "High", "Open", "Close", "Change", "Volume"]])
    #x = skp.normalize(dta_final.values[:, 4:9])
    np.set_printoptions(precision=3)
    return dta_final

def plotdta (dta_final):
    plt.plot(dta_final.Date, dta_final.Close)
    #plt.hist(dta_final.Volume, normed=True, alpha=0.9)

dta_final_scaled = dta_rescaled(dta_preparation())


def ann():
    x = dta_final_scaled.iloc[:, [2,3,4,5,6,8,9]].values        #unabhängige/exogene Variablen
    y = dta_final_scaled.iloc[:, 7].values          #abhängige/endogene/target Variable
    labelencoder_x_1 = LabelEncoder()
    x[:, 0] = labelencoder_x_1.fit_transform(x[:, 0])   #Monat -> numerisch
    labelencoder_x_2 = LabelEncoder()           #Tag -> numerisch
    x[:, 1] = labelencoder_x_2.fit_transform(x[:, 1])

    #onehotencoder = OneHotEncoder(categorical_features=[1])     #dummy-Variable für neue numerische Variablen
    #x = onehotencoder.fit_transform(x).toarray()
    #x = x[:, 1:]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
    
    return x

def rnn(dta_final):
    dta_array = np.array(dta_final.iloc[:, [7]])
    num_periods = 90    #Periodenlänge 90
    f_horizon = 1       #Forecast für eine Periode
    x_data = dta_array[:(len(dta_array)-(len(dta_array) % num_periods))]
    x_batches = x_data.reshape(-1, 90, 1)  #minus 1 Index, 80 Datensätze, 10 Spalten/Attribute je Datensatz

    y_data = dta_array[1:(len(dta_array) - (len(dta_array) % num_periods))+f_horizon]
    y_batches = y_data.reshape(-1, 90, 1)
    print(x_batches[0:2])       #zeigt zwei Batches an

    def test_data(series, forecast, num_periods):
        test_x_setup = series[-(num_periods + forecast):]    #alle Werte "num_periods+forecast" von rechts 0123456 -> 456
        testx = test_x_setup[:num_periods].reshape(-1, 90, 1)   #alle Werte von Stelle 0 bis "num_periods" von links
        testy = dta_array[-(num_periods):].reshape(-1, 90, 1)
        return testx, testy

    x_test, y_test = test_data(dta_array, f_horizon, num_periods)
    #print(x_test.shape)
    #print(x_test)

    inputs = 1
    hidden = 1000           #Anzahl an Hidden-Units
    output = 1

    x = tf.placeholder(tf.float32, [None, num_periods, inputs])
    y = tf.placeholder(tf.float32, [None, num_periods, inputs])

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.sigmoid)
    rnn_output, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)

    learning_rate = 0.001

    stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
    stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
    outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

    loss = tf.reduce_sum(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    epochs = 1000                       #Output gibt MSE alle 100 Epochs aus, dementsprechend 10x

    with tf.Session() as sess:
        init.run()
        for ep in range(epochs):
            sess.run(training_op, feed_dict={x: x_batches, y: y_batches})
            if ep % 100 == 0:
                mse = loss.eval(feed_dict={x: x_batches, y: y_batches})
                print (ep, "\tMSE:", mse)

        y_pred = sess.run(outputs, feed_dict={x: x_test})
        print(y_pred)

    plt.title("Forecast vs Actual")
    plt.plot(pd.Series(np.ravel(y_test)), "bo", markersize=10, label="Actual")
    plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=10, label="Forecast")
    plt.legend(loc="upper left")
    plt.xlabel("Time Periods")
    plt.show()

#print("Before:")
#print(dta_vanilla().head())
#print(dta_vanilla().info())
#print()
#print ("Clean data:")
#print(dta_preparation().head())
#print(dta_preparation().info())
#print()
#print("Rescaled data:")
#print(dta_rescaled(dta_preparation()).head())
#print(ann())

rnn(dta_preparation())

#plt.show(plotdta(dta_preparation()))                        #Visualisierung Rohdaten
#plt.show(plotdta(dta_rescaled(dta_preparation())))             #Visualisierung skalierte Daten