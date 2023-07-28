import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import numpy as np 
import datetime
import os
import time
import sys
from data import loadData
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM
from sklearn import metrics 
from tensorflow.keras.utils import plot_model
from contextlib import redirect_stdout

def print_model_summary(model, traj_study):
    with open(traj_study+'\\modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

def max_pos(array):
    n=len(array)
    maxi,Imaxi = array[0],0
    for i in range(n):
        elt = array[i]
        if elt>maxi:
            maxi,Imaxi=elt,i
    return maxi,Imaxi 


def output_pred(array, val):
    array = np.array(array)
    #on fait une fonction qui retourne l'augmentation maximale ou minimale de nos données en pourcentage par rapport à la    valeur de départ
    array_var_abs = (array-val)
    maxi, Imaxi = max_pos(abs(array_var_abs))
    return array[Imaxi], array_var_abs[Imaxi]

def path(path):
    if not os.path.isdir(path): # check if folder exists, otherwise create it
        os.mkdir(path)
 




## Split the time-series data into training seq X and output value Y
def extract_seqX_outcomeY(data, N, offset):
    X, y = [], []

    for i in range(offset, len(data)):
        X.append(data[i - N : i])
        y.append(data[i])

    return np.array(X), np.array(y)

def extract_seqX_outcomeY_maxi_var(data, N, offset, proj):
    X, y, affich_arr = [], [], []

    for i in range(offset, len(data)-proj):
        X.append(data[i - N : i])
        affich_arr.append(data[i - N : i+proj])
        val, diff_per = output_pred(data[i:i+proj], data[i-1])
        y.append(val)

    return np.array(X), np.array(y), np.array(affich_arr)

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) %
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def calculate_perf_metrics(var):
    ### RMSE
    rmse = calculate_rmse(
        np.array(stockprices[train_size:]["Close"]),
        np.array(stockprices[train_size:][var]),
    )
    ### MAPE
    mape = calculate_mape(
        np.array(stockprices[train_size:]["Close"]),
        np.array(stockprices[train_size:][var]),
    )
    return rmse, mape

def plot_stock_trend(var, cur_title, stockprices):
    ax = stockprices[["Close", var, "200day"]].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis("tight")
    plt.ylabel("Stock Price ($)")

def Run_LSTM(X_train, layer_units=50, couches =2, optimizer = "adam", loss = "mean_squared_error"):
    inp = Input(shape=(X_train.shape[1], 1))

    x = LSTM(units=layer_units, return_sequences=True)(inp)
    for j in range(couches-2):
        x = LSTM(units=layer_units, return_sequences=True)(x)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inp, out)

    # Compile the LSTM neural net
    model.compile(loss=loss, optimizer=optimizer) #mean_squared_error

    return model



def custom_metric(x, y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def ema(array,day_mean=5):
    n = len(array)
    alpha = 2/(day_mean+1) #on prend le alpha classique
    ema = np.zeros(n)
    ema[0] = array[0]
    for i in range(1,n):
        ema[i] = alpha*array[i] + (1-alpha)*ema[i-1]
    return ema

#on veut calculer la prediction en prenant en supposant que la tendance est la même que la tendance de la fenêtre précédente
def prediction_ema(array,val, day_gap = 1, proj = 1):
    delta = (array[-1] - array[-1-day_gap])/day_gap
    return val + delta * proj#(delta * proj)

def bonne_direction(array_test, array_X, array_Y):
    n = len(array_test)
    bon_dir = []
    for j in range(n):
        if (array_Y[j,0]-array_X[j,-1,0])*(array_test[j,0]-array_X[j,-1,0])>0:
            bon_dir.append(1.0)
        else:
            bon_dir.append(0.0)
    return np.mean(bon_dir)*100.0

def plot_loss_valloss(history, traj_study):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    fig.suptitle('Loss in training and validation', fontsize = 23, fontweight = 'bold')
    color = 'lightseagreen'
    ax1.set_xlabel('Epochs', fontsize = 20)
    ax1.set_ylabel('Train loss', color=color, fontsize = 20)
    ax1.plot(history.history['loss'], color=color, label = 'loss')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'firebrick'
    ax2.set_ylabel('Validation loss', color=color, fontsize = 20)  # we already handled the x-label with ax1
    ax2.plot(history.history['val_loss'], color=color, label = 'val_loss')
    ax1.grid(True)
    #fig.legend(fontsize = 15)
    plt.savefig(traj_study + "\\loss.png", bbox_inches='tight')
    plt.close("all")


def bonne_dire(array_test, array_X, array_Y, traj_study=None):
    n = len(array_test)
    bon_dir = np.zeros((2,2))  
    for j in range(n):
        if (array_Y[j,0]-array_X[j,-1,0])<0: #
            if (array_test[j,0]-array_X[j,-1,0])<0: #
                bon_dir[0,0]+=1
            else:
                bon_dir[0,1]+=1
        else:
            if (array_test[j,0]-array_X[j,-1,0])<0: #
                bon_dir[1,0]+=1
            else:
                bon_dir[1,1]+=1
    bon_dir = bon_dir/n
    #print("accuracy : ", np.round(np.sum(np.diag(bon_dir)) *100.0,2), "%")
    #on veut afficher les résultats sur une matrice de confusion
    if traj_study != None:
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = bon_dir, display_labels = ['short', 'long'])
        cm_display.plot()
        plt.savefig(traj_study + "\\mat_conf.png", bbox_inches='tight')
        plt.close("all")
    return bon_dir


