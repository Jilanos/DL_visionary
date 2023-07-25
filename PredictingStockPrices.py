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
    array_var_abs = abs((array-val))
    maxi, Imaxi = max_pos(array_var_abs)
    return array[Imaxi], maxi/val*100.0

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

def Run_LSTM(X_train, layer_units=50, couches =2, optimizer = "adam"):
    inp = Input(shape=(X_train.shape[1], 1))

    x = LSTM(units=layer_units, return_sequences=True)(inp)
    for j in range(couches-2):
        x = LSTM(units=layer_units, return_sequences=True)(x)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inp, out)

    # Compile the LSTM neural net
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    return model


traj = "C:\\Users\\PM263553\\Desktop\\finance\\Git\\DL_visionary\\resultat"

#je veux faire un dossier qui contienne les resultat d'un run total et donc qui me dise les paramètres utilisés
#pour ça je vais faire un dossier qui contient le nom de la fonction d'activation de la couche 1, le nom de la fonction d'activation de la couche 2, le learning rate et le nombre de points utilisés



n_points =20000 # #40000*4
form = 'h'
tf = 1


name = "minmax_0_15layers3_np_{}_tf_{}_{}".format(n_points, tf, form)
traj_study = traj + "\\study_DLV_"+name 
path(traj_study)
offset = 150

data_raw = loadData(paire="BTCBUSD", sequenceLength=int(n_points), interval_str="{}".format(tf)+form, numPartitions=4,trainProp = 0.6, validProp = 0.25, testProp  = 0.15, reload=True, ignoreTimer=offset)
data = data_raw[:,0]
data = data.reshape(-1,1)
print(np.shape(data))
#%%


test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len(data))
test_size = int(test_ratio * len(data))
print(f"train_size: {train_size}")
print(f"test_size: {test_size}")
train = data[:train_size]
test = data[train_size:]

layer_units = 15
optimizer = "adam"
cur_epochs = 15
cur_batch_size = 20
window_size = 50
proj = 6
couches = 4


# Scale our dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_data_train = scaled_data[: train.shape[0]]

# We use past 50 days’ stock prices for our training to predict the 51th day's closing price.
X_train, y_train, train_plot = extract_seqX_outcomeY_maxi_var(scaled_data_train, window_size, window_size, proj)
model = Run_LSTM(X_train, layer_units=layer_units, couches=couches, optimizer=optimizer)
model.summary()
history = model.fit(X_train, y_train, epochs=cur_epochs, batch_size=cur_batch_size, verbose=1, validation_split=0.05, shuffle=True)
plot_model(model, to_file=traj_study+'\\model_summary.png', show_shapes=True)
#%%
#on veut afficher l'évolution de la loss et val_loss avec 2 echelles différentes
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

#%%

# def preprocess_testdat(data=data, scaler=scaler, window_size=window_size, test=test):
#     raw = data[len(data) - len(test) - window_size:]
#     raw = raw.reshape(-1,1)
#     raw = scaler.transform(raw)

#     X_test = [raw[i-window_size:i, 0] for i in range(window_size, raw.shape[0])]
#     X_test = np.array(X_test)

#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#     return X_test

# X_test = preprocess_testdat()

# predicted_price_ = model.predict(X_test)
# predicted_price = scaler.inverse_transform(predicted_price_)

#%%
# Plot predicted price vs actual closing price
# plt.figure(figsize=(20, 10))
# plt.plot(predicted_price, c = 'r', label = "prédiction", marker='x')
# plt.plot(test[1:], c = 'k', label = 'price', marker='x')
# plt.grid()
# plt.legend(fontsize = 15)
# plt.show()


#%%
#on veut calculer l'exponentielle moving average d'un array
def ema(array,day_mean=5):
    n = len(array)
    alpha = 2/(day_mean+1) #on prend le alpha classique
    ema = np.zeros(n)
    ema[0] = array[0]
    for i in range(1,n):
        ema[i] = alpha*array[i] + (1-alpha)*ema[i-1]
    return ema

#on veut calculer la prediction en prenant en supposant que la tendance est la même que la tendance de la fenêtre précédente
def prediction_ema(array,day_gap = 1):
    delta = array[-1] - array[-1-day_gap]
    return array[-1] + delta

def bonne_direction(array_test, array_X, array_Y):
    n = len(array_test)
    bon_dir = []
    for j in range(n):
        if (array_Y[j,0]-array_X[j,-1,0])*(array_test[j,0]-array_X[j,-1,0])>0:
            bon_dir.append(1.0)
        else:
            bon_dir.append(0.0)
    return np.mean(bon_dir)*100.0

#%%


scaled_data_test = scaled_data[train.shape[0]:]

# We use past 50 days’ stock prices for our training to predict the 51th day's closing price.
X_test_scaled, y_test_scaled, test_plot = extract_seqX_outcomeY_maxi_var(scaled_data_test, window_size, window_size, proj)
Y_pred_scaled = model.predict(X_test_scaled)
Y_pred = scaler.inverse_transform(Y_pred_scaled)
y_test = scaler.inverse_transform(y_test_scaled)

#on veut mettre X_test_scaled sous la shape ()
X_test = scaler.inverse_transform(np.reshape(X_test_scaled, (X_test_scaled.shape[0]* X_test_scaled.shape[1],1)))
X_test = np.reshape(X_test, (X_test_scaled.shape[0], X_test_scaled.shape[1],1))
# on veut fairel a même chose pour test_plot que pour X_test_scaled
test_plot_unscaled = scaler.inverse_transform(np.reshape(test_plot, (test_plot.shape[0]* test_plot.shape[1],1)))
test_plot_unscaled = np.reshape(test_plot_unscaled, (test_plot.shape[0], test_plot.shape[1],1))


#%%
#on va faire une fonction qui détermine si l'on va dans la bonne direction
#on va faire 4 sortie : la truth augmente, la truth diminue, et ce que le modèle prédit augmente ou diminue
def bonne_dire(array_test, array_X, array_Y):
    n = len(array_test)
    bon_dir = np.zeros((2,2))  
    for j in range(n):
        if (array_Y[j,0]-array_X[j,-1,0])<0:
            if (array_test[j,0]-array_X[j,-1,0])<0:
                bon_dir[0,0]+=1
            else:
                bon_dir[0,1]+=1
        else:
            if (array_test[j,0]-array_X[j,-1,0])<0:
                bon_dir[1,0]+=1
            else:
                bon_dir[1,1]+=1
    bon_dir = bon_dir/n
    print("accuracy : ", np.round(np.mean(np.diag(bon_dir)) *100.0,2), "%")
    #on veut afficher les résultats sur une matrice de confusion
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = bon_dir, display_labels = ['short', 'long'])
    cm_display.plot()
    plt.savefig(traj_study + "\\mat_conf.png", bbox_inches='tight')
    plt.clf()
    return bon_dir
mat = bonne_dire(Y_pred, X_test, y_test)

#%%

bon_dir = []
for j in range(len(Y_pred)):
    if (Y_pred[j,0]-X_test[j,-1,0])*(y_test[j,0]-X_test[j,-1,0])>0:
        bon_dir.append(1.0)
    else:
        bon_dir.append(0.0)
print("pourcentage de bonne direction : ", np.round(np.mean(bon_dir)*100.0,2))
rmse_pred = calculate_rmse(y_test, Y_pred)
mape_pred = calculate_mape(y_test, Y_pred)
rmse_price = calculate_rmse(y_test, X_test[:,-1,:])
mape_price = calculate_mape(y_test, X_test[:,-1,:])

print("Same Price : bonne direction  : \t\t", bonne_direction(y_test, X_test, y_test))
print("Same Price : rmse : \t\t", rmse_price)
print("Same Price : mape : \t\t", mape_price, '\n')
print("Prédiction LTSM : bonne direction  : \t\t", bonne_direction(y_test, X_test, Y_pred))
print("Prédiction LTSM :rmse : \t\t", rmse_pred)
print("Prédiction LTSM : mape : \t\t", mape_pred, '\n')
plt.figure(figsize=(20, 10))

n_plot = 50
pas = test_size//n_plot
plt.figure(figsize=(20, 10))
for j in range(0, n_plot):
    k = j*pas
    #on affiche maintenant
    plt.clf()
    plt.plot(test_plot_unscaled[k], c = 'k', label = 'price')
    plt.plot([window_size-1,window_size+proj],[X_test[k,window_size-1,0],Y_pred[k,0]], c = 'r', label = 'prédiction')
    plt.scatter(window_size+proj,Y_pred[k,0], c = 'blue', label = 'prediction', marker='x')
    plt.scatter(window_size+proj,y_test[k,0], c = 'red', label = 'price', marker='x')
    plt.title("prédiction du prix à t+1", fontsize = 20)
    plt.xlabel("temps", fontsize = 20)
    plt.ylabel("prix", fontsize = 20)
    plt.grid(True)
    plt.legend(fontsize = 15)
    plt.savefig(traj_study + "\\plot_TEST_{}.png".format(k))
plt.close("all")

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

