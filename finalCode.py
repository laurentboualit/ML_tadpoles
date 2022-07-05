# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:20:03 2022

@author: d4kro
"""

#%%-----------0. loading package-----------------------------------------------

import pandas as pd
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from matplotlib import pyplot

from scipy.stats import randint as randint

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import SGD

label_encoder = preprocessing.LabelEncoder()
#%%-----------1. loading data--------------------------------------------------

# import data as dataframe
df = pd.read_csv('Master_day6.csv',low_memory=False)

#%%-----------2. data processing-----------------------------------------------

# drop the raw with concentration value of DMSO (solute test)
df = df[df["Nom_Conc"].str.contains("DMSO") == False]

# change type string to numeric for "conc_name" column
df['Nom_Conc'] = pd.to_numeric(df['Nom_Conc'])

df.info() # checking the columns

# keeping only the needed columns
data = df[['Area','Perimeter','Major','Minor','step_length','step_speed',
           'abs_angle','rel_angle','Nom_Conc']].copy()
data.info()# checking the new columns

#%%-----------3. take 5 random line & export it (no use for the code)----------

#sampleDF = data.sample(n = 5)
#sampleDF.to_csv('sampleDF.csv')

#%%-----------4. delet infinite and nan values---------------------------------

# turn inf values in nan
data.replace([np.inf, -np.inf], np.nan, inplace=True)
# check that no raw has NaN values
data.isna().values.any()
# if true -> deleting all raw containing NaN
data_noNAN = data.dropna(axis=0)
# check again that no raw has NaN values
data_noNAN.isna().values.any()

#%%-----------5. represent graphicaly data-------------------------------------

# select two variables of interest and the concentration
X1 = data_noNAN[['Major']].copy()
X2 = data_noNAN[['step_speed']].copy()
c = data_noNAN[['Nom_Conc']].copy()
#encode the concentration
c = label_encoder.fit_transform(c) 

# plot the points
fig = plt.figure()
plt.scatter(X1, X2, c=c)

# Set figure title and axis labels
plt.title('step_speed vs abs_angle for each measurement point')
plt.xlabel("Major [pixel]")
plt.ylabel("step_speed [pixel]")

#%%-----------6. split the data------------------------------------------------
# select the variables
X = data_noNAN[['step_speed','Major','Minor','step_length']].copy()
Y = data_noNAN[['Nom_Conc']].copy()

# split the dataset in train, validation and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7)
X_test, X_val, y_test, y_val = train_test_split(X, Y, test_size=0.15, train_size=0.15)

# label encoding in order to normalise the target variable
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

#checking the length of each dataset
print('length of train set is :')
print(len(X_train))
print('length of test set is :')
print(len(X_test))
print('length of validation set is :')
print(len(X_val))
#%%-----------7. RandomForestClassifier----------------------------------------

#%%--------------7.1 train and test the model (RFC & default HP)---------------
# set the classifier
rfc = RandomForestClassifier(n_jobs=-1)

# fit the data (training)
rfc.fit(X_train, y_train)

# predict after training on test set
RFC_test = rfc.predict(X_test)

# print the matrix and the accuracy
print('\nMatrix confusion for RandomForestClassifier (Default HP) :')
print(confusion_matrix(y_test, RFC_test))

acc_RFC = accuracy_score(y_test, RFC_test)
print(f'\nThe accuracy of the model RandomForestClassifier is (Default HP) {acc_RFC:.1%}')
#%%--------------7.2 search HP for RFC-----------------------------------------

# define HP for to search
params = { 
            'n_estimators': [80,120],
            'max_features': ['sqrt','sqrt',None],
            'criterion' :['gini', 'entropy','log_loss']
         }
#define the Gridsearch
gsc = GridSearchCV(rfc, params, cv=5,n_jobs=-1)

#fit with the Gridsearch
gsc.fit(X_train, y_train)

#Results key
print('\nResults keys :')
sorted(gsc.cv_results_.keys())

# print the best hyperparameters
print('\nBest params :')
print(gsc.best_params_)

# and score
print('\nBest score :')
print(gsc.best_score_)

#Best estimator
print('\nBest estimator :')
gsc.best_estimator_

#%%--------------7.3 train and test the model (RFC & HP)-----------------------

# Since no HP where found to optimize the model (see section 7.2), nothing is 
# written in this part. Look at the 7.4 confusion matrix section to see the 
# results. 

#%%--------------7.4 plot the confusion matrix---------------------------------
plot_confusion_matrix(rfc, X_test, y_test)  
plt.show() 
#%%-----------8. DecisionTreesClassifier---------------------------------------

#%%--------------8.2 search HP for DTC-----------------------------------------
# define HP to search
parameters = {
              'criterion' : ['gini','entropy'],
              'splitter' : ['best','random'],
              "max_depth": [25,50,75,100,None],
              'min_samples_split':[2,3,4,5],
              'min_samples_leaf':[1,2,3,4],
              'min_weight_fraction_leaf':[0,0.5],
              'max_features':[1,2,'auto','sqrt','log2',None]
             }
#define the Gridsearch
gsc1 = GridSearchCV(dtc, parameters,cv=5, n_jobs=3)

# fit the data (training)
gsc1.fit(X_train,y_train)

#Results key
print('\nResults keys :')
sorted(gsc1.cv_results_.keys())

# print the best hyperparameters
print('\nBest params :')
print(gsc1.best_params_)

# and score
print('\nBest score :')
print(gsc1.best_score_)

#Best estimator
print('\nBest estimator :')
gsc1.best_estimator_

#%%--------------8.3 search HP for DTC (random)--------------------------------
    
# define HP to random search
param_dist = {"max_depth": [25,50,75,100,None],
              'min_samples_split':[2,3,4,5],
              'min_samples_leaf':[1,2,3,4],
              'min_weight_fraction_leaf':[0,0.5],
              'max_features':[1,2,'auto','sqrt','log2',None]
              }
#define the Randomizedsearch
gsc_rand = RandomizedSearchCV(dtc, param_dist, cv=5,n_jobs=-1)

# fit the data (training)
gsc_rand.fit(X_train,y_train)

# print the results
print(" Results from Random Search " )
print("\n The best estimator across ALL searched params:\n", gsc_rand.best_estimator_)
print("\n The best score across ALL searched params:\n", gsc_rand.best_score_)
print("\n The best parameters across ALL searched params:\n", gsc_rand.best_params_)
#%%--------------8.4 train and test the model (DTC & HP)-----------------------

# Since no HP where found to optimize the model (see section 7.2 and 8.3), 
# nothing is written in this part. 

#%%-----------9. Softmax regression--------------------------------------------

#https://towardsdatascience.com/softmax-regression-in-python-multi-class-classification-3cb560d90cb2
#https://awjuliani.medium.com/simple-softmax-in-python-tutorial-d6b4c4ed5c16#:~:text=Softmax%20regression%20is%20a%20method,any%20number%20of%20possible%20classes.
#https://towardsdatascience.com/multiclass-classification-with-softmax-regression-explained-ea320518ea5d

#%%--------------9.1 One-hot encoding------------------------------------------

def one_hot(y, c_length):
    
    # y--> y_train/test/val
    # c--> Number of classes.
    
    # A zero matrix of size (m, c)
    y_hot = np.zeros((len(y), c_length))
    
    # Putting 1 for column where the label is,
    # Using multidimensional indexing.
    y_hot[np.arange(len(y)), y] = 1
    
    return y_hot

#%%--------------9.2 Softmax function------------------------------------------

def softmax(z):
    
    # z--> linear part.
    
    # subtracting the max of z for numerical stability.
    z = z/(np.max(z)/16)
    exp = np.exp(z)
    
    # Calculating softmax for all examples.
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
        
    return exp

#%%--------------9.4 Training--------------------------------------------------

def fit(X, y, lr, c, epochs):
    
    # X --> Input.
    # y --> true/target value.
    # lr --> Learning rate.
    # c --> Number of classes.
    # epochs --> Number of iterations.
    
        
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Initializing weights and bias randomly.
    w = np.random.random((n, c))
    b = np.random.random(c)
    # Empty list to store losses.
    losses = []
    
    # Training loop.
    for epoch in range(epochs):
        
        # Calculating hypothesis/prediction.
        z = np.dot(X,w) + b
        
        y_hat = softmax(z)
        
        # One-hot encoding y.
        y_hot = one_hot(y, c)
        
        # Calculating the gradient of loss w.r.t w and b.
        w_grad = (1/m)*np.dot(X.T, (y_hat - y_hot)) 
        b_grad = (1/m)*np.sum(y_hat - y_hot)
        
        # Updating the parameters.
        w = w - lr*w_grad
        b = b - lr*b_grad
        
        # Calculating loss and appending it in the list.
        loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
        losses.append(loss)
        # Printing out the loss at every 100th iteration.
        #if epoch%100==0:
        print('Epoch {epoch}==> Loss = {loss}'
                  .format(epoch=epoch, loss=loss))
    return w, b, losses

#%%--------------9.5 train data------------------------------------------------
w, b, l = fit(X_train, y_train, lr=1.5, c=5, epochs=20)

#%%--------------9.6 plot the loss function------------------------------------

plt.plot(l)
plt.ylabel('Log Loss')
plt.xlabel('Iterations')
plt.title('Loss Function Graph')

#%%--------------9.7 Predict & measure accuracy--------------------------------
def predict(X, w, b):
    
    # X --> Input.
    # w --> weights.
    # b --> bias.
    
    # Predicting
    z = np.dot(X,w) + b
    y_hat = softmax(z)
    
    # Returning the class with highest probability.
    return np.argmax(y_hat, axis=1)

def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)

#%%--------------9.8 compute the accuracies------------------------------------

train_p = predict(X_train, w, b)
print(accuracy(y_train, train_p))

# Accuracy for test set.
# Flattening and normalizing.
test_p = predict(X_test, w, b)
print(accuracy(y_test, test_p))

#%%----------10. Multi-Class Classification Loss Functions---------------------

#%%-------------10.1 define model----------------------------------------------

model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(..., activation='softmax'))

#%%-------------10.2 compile model---------------------------------------------
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='...', optimizer=opt, metrics=['accuracy'])

#%%-------------10.3 fit model-------------------------------------------------

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    epochs=100, verbose=0)

#%%-------------10.4 evaluate the model----------------------------------------

train_acc = model.evaluate(X_train, y_train, verbose=0)
test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#%%-------------10.4 plot the results------------------------------------------
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

#%%----------11. Apply the model selected on day 7-----------------------------
#%%-------------11.1 data preprocessing----------------------------------------
# import data as dataframe
df7 = pd.read_csv('Master_day7.csv',low_memory=False)

# drop the raw with concentration value of DMSO (solute test)
df7 = df7[df7["Nom_Conc"].str.contains("DMSO") == False]

# change type string to numeric for "conc_name" column
df7['Nom_Conc'] = pd.to_numeric(df7['Nom_Conc'])

df7.info() # checking the columns

# keeping only the needed columns
data7 = df7[['Area','Perimeter','Major','Minor','step_length','step_speed',
           'abs_angle','rel_angle','Nom_Conc']].copy()
data7.info()# checking the new columns

# turn inf values in nan
data7.replace([np.inf, -np.inf], np.nan, inplace=True)
# check that no raw has NaN values
data7.isna().values.any()
# if true -> deleting all raw containing NaN
data_noNAN7 = data.dropna(axis=0)
# check again that no raw has NaN values
data_noNAN7.isna().values.any()

#%%-------------11.2 prepare the data------------------------------------------

# select the variables
X7 = data_noNAN7[['step_speed','Major','Minor','step_length']].copy()
Y7 = data_noNAN7[['Nom_Conc']].copy()

# label encoding in order to normalise the target variable
label_encoder = preprocessing.LabelEncoder()
Y7 = label_encoder.fit_transform(Y7)

#%%-------------12.3 Use the model ont the 7 day data--------------------------

# predict after training on test set
RFC7 = rfc.predict(X7)

# print the matrix and the accuracy
print('\nMatrix confusion for RandomForestClassifier for 7 days data (Default HP) :')
print(confusion_matrix(Y7, RFC7))

acc_RFC7 = accuracy_score(Y7, RFC7)
print(f'\nThe accuracy of the model RandomForestClassifier for 7 days data is (Default HP) {acc_RFC7:.1%}')
#%%----------12. Apply the model selected on day 8-----------------------------
#%%-------------11.3 Use the model ont the 7 day data--------------------------
#%%----------12. Apply the model selected on day 8-----------------------------

# predict after training on test set
RFC_test7 = rfc.predict(X_test7)

# print the matrix and the accuracy
print('\nMatrix confusion for RandomForestClassifier for 7 days data (Default HP) :')
print(confusion_matrix(y_test7, RFC_test7))

acc_RFC = accuracy_score(y_test7, RFC_test7)
print(f'\nThe accuracy of the model RandomForestClassifier for 7 days data is (Default HP) {acc_RFC:.1%}')
#%%-------------12.1 data preprocessing----------------------------------------
# import data as dataframe
df8 = pd.read_csv('Master_day8.csv',low_memory=False)

# drop the raw with concentration value of DMSO (solute test)
df8 = df8[df8["Nom_Conc"].str.contains("DMSO") == False]

# change type string to numeric for "conc_name" column
df8['Nom_Conc'] = pd.to_numeric(df8['Nom_Conc'])

df8.info() # checking the columns

# keeping only the needed columns
data8 = df8[['Area','Perimeter','Major','Minor','step_length','step_speed',
           'abs_angle','rel_angle','Nom_Conc']].copy()
data8.info()# checking the new columns

# turn inf values in nan
data8.replace([np.inf, -np.inf], np.nan, inplace=True)
# check that no raw has NaN values
data8.isna().values.any()
# if true -> deleting all raw containing NaN
data_noNAN8 = data.dropna(axis=0)
# check again that no raw has NaN values
data_noNAN8.isna().values.any()

#%%-------------12.2 prepare the data------------------------------------------

# select the variables
X8 = data_noNAN8[['step_speed','Major','Minor','step_length']].copy()
Y8 = data_noNAN8[['Nom_Conc']].copy()

# label encoding in order to normalise the target variable
label_encoder = preprocessing.LabelEncoder()
Y8 = label_encoder.fit_transform(Y8)

#%%-------------12.3 Use the model ont the 8 day data--------------------------

# predict after training on test set
RFC8 = rfc.predict(X8)

# print the matrix and the accuracy
print('\nMatrix confusion for RandomForestClassifier for 8 days data (Default HP) :')
print(confusion_matrix(Y8, RFC8))

acc_RFC8 = accuracy_score(Y8, RFC8)
print(f'\nThe accuracy of the model RandomForestClassifier for 8 days data is (Default HP) {acc_RFC8:.1%}')

plot_confusion_matrix(rfc, X8, Y8)  
plt.show() 

























