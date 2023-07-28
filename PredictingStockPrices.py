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
from fonctions import *
from contextlib import redirect_stdout
 
traj00 = "C:\\Users\\PM263553\\Desktop\\finance\\Git\\DL_visionary\\resultat"
traj = traj00 + "\\multi_test_%4"
path(traj)


#données à télécharger
n_points =15000 # #40000*4
form = 'h'
tf = 1
interv_form = "{}".format(tf)+form

#règles des X et Y
window_size = 40
proj = 4
test_ratio = 0.2
training_ratio = 1 - test_ratio

#paramètres du modèle
layer_units = 15
cur_epochs = 15
cur_batch_size = 20
couches = 4
loss = "mean_squared_error" #mean_absolute_percentage_error #mean_squared_error 
verbose = 0
learning_rate = 0.001


#dictionnaire des résultats
res_dic = {}
for interv_form in [ "5m"]:  #"1h", "15m",

    data_raw = loadData(paire="BTCBUSD", sequenceLength=int(n_points), interval_str=interv_form, numPartitions=4, reload=True)
    data = data_raw[:,0]
    data = data.reshape(-1,1)

    train_size = int(training_ratio * len(data))
    test_size = int(test_ratio * len(data))
    train = data[:train_size]
    test = data[train_size:]

    # Scale our dataset
    scaler = StandardScaler()
    scaled_data_train = scaler.fit_transform(train)
    scaled_data_test = scaler.transform(test)

    # We use past 50 days’ stock prices for our training to predict the 51th day's closing price.
    X_train, y_train, train_plot = extract_seqX_outcomeY_maxi_var(scaled_data_train, window_size, window_size, proj)
    X_test_scaled, y_test_scaled, test_plot = extract_seqX_outcomeY_maxi_var(scaled_data_test, window_size, window_size, proj)
    train_size, test_size = np.shape(X_train)[0], np.shape(X_test_scaled)[0]
    #on veut mettre X_test_scaled sous la shape ()
    X_test = np.reshape(scaler.inverse_transform(np.reshape(X_test_scaled, (X_test_scaled.shape[0]* X_test_scaled.shape[1],1))), (X_test_scaled.shape[0], X_test_scaled.shape[1],1))

    test_plot_unscaled = np.reshape(scaler.inverse_transform(np.reshape(test_plot, (test_plot.shape[0]* test_plot.shape[1],1))), (test_plot.shape[0], test_plot.shape[1],1))
    y_test = scaler.inverse_transform(y_test_scaled)
    #on va prédire avec 2 ema les resultats :
    ema_pred = np.zeros((test_size,1)) 
    up_list, down_list = 0, 0
    #y_test = y_test + X_test[:,-1,:]
    for i in range(test_size):
        ema_1 = ema(X_test[i])
        pred_ema = prediction_ema(ema_1, X_test[i,-1], day_gap = 2, proj=proj)
        ema_pred[i] = pred_ema
        if y_test[i]>X_test[i,-1]:
            up_list +=1
        else:
            down_list  +=1

    bon_dir_ema = bonne_dire(ema_pred, X_test, y_test)
    rmse_price = calculate_rmse(y_test, X_test[:,-1,:])
    mape_price = calculate_mape(y_test, X_test[:,-1,:])
    rmse_ema = calculate_rmse(y_test, ema_pred)
    mape_ema = calculate_mape(y_test, ema_pred)
    print("Same Price : rmse : \t\t", rmse_price)
    print("Same Price : mape : \t\t", mape_price, '\n')
    print("Prédiction EMA : bonne direction  : \t\t", np.round(np.sum(np.diag(bon_dir_ema)) *100.0,2))
    print("Prédiction EMA : rmse : \t\t", rmse_ema)
    print("Prédiction EMA : mape : \t\t", mape_ema, '\n')
    
    name_ema = "{}_EMA".format(interv_form)
    res_dic[name_ema] = {}
    res_dic[name_ema]["bon_dir"] = np.sum(np.diag(bon_dir_ema)) *100.0
    res_dic[name_ema]["rmse"] = rmse_ema
    res_dic[name_ema]["mape"] = mape_ema
    res_dic[name_ema]["ratio_up_down"] = np.sum(bon_dir_ema[:,1])/np.sum(bon_dir_ema[:,0])
    
    name_same = "{}_same".format(interv_form)
    res_dic[name_same] = {}
    res_dic[name_same]["bon_dir"] = 50.0
    res_dic[name_same]["rmse"] = rmse_price
    res_dic[name_same]["mape"] = mape_price
    res_dic[name_same]["ratio_up_down"] = up_list/down_list



    for learning_rate in [  0.0001, 0.00001]: #
        for layer_units in [10,20]:    #, 20, 30
            for couches in [2,4]:     #, 3, 4
                print("\nModèle avec {} couches de {} layers sur la tf {} et learning rate de {}".format(couches, layer_units, interv_form, learning_rate))
                name_dic = "{}_{}_{}_{}".format(interv_form, layer_units, couches, learning_rate)
                res_dic[name_dic] = {}

                name = "minmax_0_{}layers{}_np_{}_tf_{}_{}".format(layer_units, couches, n_points, interv_form, learning_rate)
                traj_study = traj + "\\study_DLV_"+name 
                path(traj_study)
                optimizer = Adam(learning_rate=learning_rate)
                model = Run_LSTM(X_train, layer_units=layer_units, couches=couches, optimizer=optimizer, loss=loss)
                print_model_summary(model, traj_study)
                plot_model(model, to_file=traj_study+'\\model_summary.png', show_shapes=True)
                
                
                history = model.fit(X_train, y_train, epochs=cur_epochs, batch_size=cur_batch_size, verbose=verbose, validation_split=0.05, shuffle=True)
                plot_loss_valloss(history, traj_study)

                Y_pred_scaled = model.predict(X_test_scaled, verbose=0)
                Y_pred = scaler.inverse_transform(Y_pred_scaled)
                # Y_pred_ok = scaler.inverse_transform((X_test_scaled[:,-1,:])+Y_pred_scaled)
                # Y_test_Ok = scaler.inverse_transform((X_test_scaled[:,-1,:])+y_test_scaled)
                # pred_ema_ok = (X_test[:,-1,:]+ema_pred)

                

                bon_dir_pred = bonne_dire(Y_pred, X_test, y_test, traj_study)

                rmse_pred = calculate_rmse(y_test, Y_pred)
                mape_pred = calculate_mape(y_test, Y_pred)
            

                print("Prédiction LTSM : bonne direction  : \t\t",  np.round(np.sum(np.diag(bon_dir_pred)) *100.0,2),)
                print("Prédiction LTSM :rmse : \t\t", rmse_pred)
                print("Prédiction LTSM : mape : \t\t", mape_pred, '\n')


                res_dic[name_dic]["bon_dir"] = np.sum(np.diag(bon_dir_pred)) *100.0
                res_dic[name_dic]["rmse"] = rmse_pred
                res_dic[name_dic]["mape"] = mape_pred
                res_dic[name_dic]["ratio_up_down"] = np.sum(bon_dir_pred[:,1])/np.sum(bon_dir_pred[:,0])

                
                n_plot = 50
                pas = test_size//n_plot
                plt.figure(figsize=(20, 10))
                for j in range(0, n_plot):
                    k = j*pas
                    #on affiche maintenant
                    plt.clf()
                    plt.plot(test_plot_unscaled[k], c = 'k', label = 'price')
                    plt.plot([window_size-1,window_size+proj-1],[X_test[k,window_size-1,0],Y_pred[k,0]], c = 'forestgreen', label = 'Prédiction LSTM')
                    plt.plot([window_size-1,window_size+proj-1],[X_test[k,window_size-1,0],ema_pred[k,0]], c = 'purple', label = 'Prédiction EMA')
                    plt.plot([window_size-1,window_size+proj-1],[X_test[k,window_size-1,0],X_test[k,window_size-1,0]], c = 'k', label = 'Same price', linestyle = '--')
                    #plt.scatter(window_size+proj,Y_pred[k,0], c = 'green', label = 'Prédiction', marker='x')
                    plt.scatter(window_size+proj-1,y_test[k,0], c = 'maroon', label = 'Truth', marker='x', s = 200, linewidths=4)
                    plt.axvspan(0, window_size-1, facecolor='lightskyblue' , alpha=0.15)
                    plt.axvspan(window_size-1, window_size+proj-1, facecolor='burlywood' , alpha=0.15)
                    plt.title("Prédiction du max du prix à t+{}".format(proj) , fontsize = 22, fontweight = 'bold')
                    plt.xlabel("Timeframe", fontsize = 20)
                    plt.xlim(0, window_size+proj-0.5)
                    plt.ylabel("Prix", fontsize = 20)
                    plt.xticks(fontsize = 15)
                    plt.yticks(fontsize = 15)
                    plt.grid(True)
                    plt.legend(fontsize = 15)
                    plt.savefig(traj_study + "\\plot_TEST_{}.png".format(k))
                plt.close("all")


#%%
#on veut plot les résultats
#on veut plot les directions predite d'abord
decimal_key = {'bon_dir': 3,  'rmse': 2,  'mape': 4, 'ratio_up_down' : 3}
for key, elt  in res_dic.items():
    string = ""
    for nom, val in elt.items():
        string += "\t" + nom + " = "  + str(np.round(val,decimals=decimal_key[nom] ) ) + ", "
    print(key + " : " + string)


sys.exit()






#%%
n_plot = 250
n_calc = len(test)-2 * window_size
X_array = np.zeros((n_calc, window_size, 1))
X_array_noscale = np.zeros((n_calc, window_size, 1))
Y_array = np.zeros((n_calc, 1))
Y_ema_1 = np.zeros((n_calc, 1))
Y_ema_3 = np.zeros((n_calc, 1))
for j in range(0, n_calc):
    X = test[j:j+window_size]
    X_ema = ema(X)
    Y_ema_1[j] = prediction_ema(X_ema,day_gap = 1)
    Y_ema_3[j] = prediction_ema(X_ema,day_gap = 3)
    X_noscale = X.reshape(-1,1)
    X_array[j] = scaler.transform(X_noscale)
    X_array_noscale[j] = X_noscale
    Y_array[j] = test[j+window_size]
Y_pred = model.predict(X_array)
Y_pred = scaler.inverse_transform(Y_pred)

bon_dir = []
for j in range(n_calc):
    if (Y_pred[j,0]-X_array_noscale[j,-1,0])*(Y_array[j,0]-X_array_noscale[j,-1,0])>0:
        bon_dir.append(1.0)
    else:
        bon_dir.append(0.0)
print("pourcentage de bonne direction : ", np.round(np.mean(bon_dir)*100.0,2))
rmse_pred = calculate_rmse(Y_array, Y_pred)
mape_pred = calculate_mape(Y_array, Y_pred)
rmse_price = calculate_rmse(Y_array, X_array_noscale[:n_calc,-1,:])
mape_price = calculate_mape(Y_array, X_array_noscale[:n_calc,-1,:])
rmse_ema_1 = calculate_rmse(Y_array, Y_ema_1)  
mape_ema_1 = calculate_mape(Y_array, Y_ema_1)
rmse_ema_3 = calculate_rmse(Y_array, Y_ema_3)
mape_ema_3 = calculate_mape(Y_array, Y_ema_3)
print("Same Price : bonne direction  : \t\t", bonne_direction(Y_array, X_array_noscale, Y_array))
print("Same Price : rmse : \t\t", rmse_price)
print("Same Price : mape : \t\t", mape_price, '\n')
print("Prédiction LTSM : bonne direction  : \t\t", bonne_direction(Y_array, X_array_noscale, Y_pred))
print("Prédiction LTSM :rmse : \t\t", rmse_pred)
print("Prédiction LTSM : mape : \t\t", mape_pred, '\n')
print("Prédiction EMA 1 : bonne direction  : \t\t", bonne_direction(Y_array, X_array_noscale, Y_ema_1))
print("Prédiction EMA 1 : rmse : \t\t", rmse_ema_1)
print("Prédiction EMA 1 : mape : \t\t", mape_ema_1, '\n')
print("Prédiction EMA 3 : bonne direction  : \t\t", bonne_direction(Y_array, X_array_noscale, Y_ema_3))
print("Prédiction EMA 3 : rmse : \t\t", rmse_ema_3)
print("Prédiction EMA 3 : mape : \t\t", mape_ema_3, '\n')

plt.figure(figsize=(20, 10))
for j in range(0, n_plot):
    #on affiche maintenant
    plt.clf()
    plt.plot(X_array_noscale[j,:,0], c = 'k', label = 'price')
    plt.plot([window_size-1,window_size],[X_array_noscale[j,window_size-1,0],Y_array[j,0]], c = 'r', label = 'prédiction')
    plt.scatter(window_size,Y_pred[j,0], c = 'blue', label = 'prediction', marker='x')
    plt.scatter(window_size,Y_array[j,0], c = 'red', label = 'price', marker='x')
    plt.title("prédiction du prix à t+1", fontsize = 20)
    plt.xlabel("temps", fontsize = 20)
    plt.ylabel("prix", fontsize = 20)
    plt.grid(True)
    plt.legend(fontsize = 15)
    plt.savefig(traj_study + "\\plot_TEST_{}.png".format(j))
plt.close("all")


#%%
print("Test sur les données de TRAIN")
n_plot = 250
n_calc = 2000
X_array = np.zeros((n_calc, window_size, 1))
X_array_noscale = np.zeros((n_calc, window_size, 1))
Y_array = np.zeros((n_calc, 1))
for j in range(0, n_calc):
    X = train[j:j+window_size]
    X_noscale = X.reshape(-1,1)
    X_array[j] = scaler.transform(X_noscale)
    X_array_noscale[j] = X_noscale
    Y_array[j] = train[j+window_size]
Y_pred = model.predict(X_array)
Y_pred = scaler.inverse_transform(Y_pred)

bon_dir = []
for j in range(n_calc):
    if (Y_pred[j,0]-X_array_noscale[j,-1,0])*(Y_array[j,0]-X_array_noscale[j,-1,0])>0:
        bon_dir.append(1.0)
    else:
        bon_dir.append(0.0)
print("pourcentage de bonne direction : ", np.round(np.mean(bon_dir)*100.0,2))
rmse_pred = calculate_rmse(Y_array, Y_pred)
print("Prédiction rmse : ", rmse_pred)
rmse_price = calculate_rmse(Y_array, X_array_noscale[:n_calc,-1,:])
print("Prix rmse : ", rmse_price)

plt.figure(figsize=(20, 10))
for j in range(0, n_plot):
    #on affiche maintenant
    plt.clf()
    plt.plot(X_array_noscale[j,:,0], c = 'k', label = 'price')
    plt.plot([window_size-1,window_size],[X_array_noscale[j,window_size-1,0],Y_array[j,0]], c = 'r', label = 'prédiction')
    plt.scatter(window_size,Y_pred[j,0], c = 'blue', label = 'prediction', marker='x')
    plt.scatter(window_size,Y_array[j,0], c = 'red', label = 'price', marker='x')
    plt.title("TRAIN : prédiction du prix à t+1", fontsize = 20)
    plt.xlabel("temps", fontsize = 20)
    plt.ylabel("prix", fontsize = 20)
    plt.grid(True)
    plt.legend(fontsize = 15)
    plt.savefig(traj_study + "\\plot_train_{}.png".format(j))
plt.close("all")

