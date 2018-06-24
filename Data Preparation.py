import csv
import pandas as pd
import numpy as np
import json
from pprint import pprint
import requests
import seaborn as sns
import matplotlib.pyplot as plt
#import string as str
import re
#from sklearn.neural_network import MPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow


def helloWorld (x):
 return x

print (helloWorld("Hello World"))

a = 12
b = a

b = 1

print(a)
print(b)

csv_file = open("G:/Desktop/Uni/Master/5. Semester/Applied Data Science - Case Studies/BrentDataset.csv")

line = next(csv_file)       #get next line in file
print(line)

line2 = next(csv_file)
print(line2)



reader = csv.reader(csv_file)



headers = next(reader)
pprint(headers)
line = next(reader)
pprint(line)

dialect = csv.Sniffer().sniff(csv_file.read(1024))

                        #json loads -> load data in dictionary

pandas = pd.read_csv("G:/Desktop/Uni/Master/5. Semester/Applied Data Science - Case Studies/BrentDataset.csv")

print(pandas)
print()
#DataFrames

dta = pd.read_csv("G:/Desktop/Uni/Master/5. Semester/Applied Data Science - Case Studies/BrentDataset.csv")

print(dta.index)
print()
print(dta.columns)
print()
print(dta.head())               #head return first 5 rows

dta = dta.set_index("Date")         #Date jetzt Index, keine Spalte mehr
print()
print(dta.index)

print()
print(dta.Low)

print(dta[["Low", "Vol."]])

print(dta.loc[[2018]])              #Label based indexing -> zugehörige Datensätze zum Index

print(dta.iloc[[0,5]])               #Integer based indexing -> Datensätze 1 und 6


#Cleaning Data for Types

print(dta.dtypes[["Low", "Vol.", "Change %"]])

print(dta.info())
                            # del.

#dta = pd.read_csv("G:/Desktop/Uni/Master/5. Semester/Applied Data Science - Case Studies/BrentDataset.csv",
#                 dtype ={
#                      "Data": "datetime64",
#                      "Vol.": "float64",
#                     "Change %": "float64"
#                 }
#                  )
                        #lambda functions, anonymous function, just local
                        #delete column: del dta["Change %"]

dta = pd.read_csv("G:/Desktop/Uni/Master/5. Semester/Applied Data Science - Case Studies/BrentDataset.csv")
print(dta.info())

dta_withIndex = dta.reset_index()

print(dta_withIndex.head())        #create new index
print(dta_withIndex.info())

#dta_splited = dta_withIndex.str.split
#print(dta_splited)

date_split = dta_withIndex["index"].str.split(expand=True)

print(date_split.head())

del dta_withIndex["index"]

dta_split = pd.concat((date_split, dta_withIndex), axis=1)         #Price=Close; Change is (Price_today/Price_yesterday)-1
print(dta_split.head())

dta_split = dta_split.rename(columns={0: "Month", 1: "Day", "Date": "Year", "Price": "Close", "Vol.": "Volume", "Change %": "Change"})
print(dta_split.head())

#dta_split_s = dta_split.Change.str.strip("%")
#dta_split_t = dta_split.Change.apply(lambda x: x.strip("%"))

dta_split.Change = [x.strip("%") for x in dta_split.Change]

print(dta_split.head())

print(dta_split.dtypes)

dta_split.Volume = [x.replace("-","0") for x in dta_split.Volume]
dta_split[["Volume"]] = dta_split[["Volume"]].astype(float)
dta_split[["Change"]] = dta_split[["Change"]].astype(float)

#print(dta_split.Volume.head())
#print(dta_split.Volume.str.endswith("-").any())
#grouped = dta_split.groupby("Volume")
print(dta_split.head())
print(dta_split.dtypes)

#spalte aus jahr-monat-tag -> datentyp datum und als index
#sortieren nach datum aufsteigend
#funktionen nutzen und code sortieren
#ausreißer entdecken
#neue spalte "intradayvolatility" zur abdeckung |high-low|
#neue spalte "avg" zur abgedeckung durchschnitt open,low,close
#einfaches ann auf daten anwenden (target ist close/price)

dta_split[["Year"]] = dta_split[["Year"]].astype(str)
print(dta_split.dtypes)

dta_split["Date"] = dta_split["Year"] + "/" + dta_split["Month"] + "/" + dta_split["Day"]
print(dta_split.head())

dta_date = pd.to_datetime(dta_split["Date"])
del dta_split["Date"]
dta_final = pd.concat((dta_split, dta_date), axis=1)

print (dta_final.head())
print(dta_final.dtypes)

#print(dta_final.iloc[:,9].head())
#dta_final.drop(dta_final.columns[10], axis=1, inplace=True)      #ohne inplace wäre extra Zuweisung nötig -> dta_final = ...
#print(dta_final.head())

dta_final.sort_values(by=["Date"], ascending=True, inplace=True)

print(list(dta_final.columns.values))
dta_final = dta_final[["Date","Year","Month","Day","Low","High","Open","Close","Change","Volume"]]
print(dta_final.head())


#sns.distplot(dta_final.Volume.dropna())
#plt.show()

#sns.tsplot(dta_final.Close, time="Date", unitcolor="green")
#sns.set()
plt.plot(dta_final.Date, dta_final.Close)
plt.show()

plt.hist(dta_final.Volume, normed=True, alpha=0.9)
plt.show()

#sns.factorplot(dta_final.Date, data=dta_final.Volume, aspect=2)
#plt.show()

#splitting dataset into training and test


#x = dta_final.iloc[:, [1,2,3,4,5,6,8,9]]        #unabh. variablen
#x = dta_final.iloc[:, [4,5,6,8,9]]
#y = dta_final.iloc[:, 7]                        #target

#print(y.head())

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_train = sc.transform(x_train)

#classifier = Sequential()

#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 5))

#classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 10)

#print(classifier.predict(x_test))


csv_file.close()

