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
from tensorflow.keras.utils import plot_model

def path(path):
    if not os.path.isdir(path): # check if folder exists, otherwise create it
        os.mkdir(path)
 
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
    array_percent = (array-val)/val
    if np.max(array_percent) > np.abs(np.min(array_percent)):
        return np.max(array_percent)
    else:
        return np.min(array_percent)
    
    
def custom_loss(y_true, y_pred):
    # Calcul de la MSE
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Calcul de la loss doublée lorsque la prédiction et la vraie valeur sont de signe opposé
    mask = tf.math.multiply(y_true, y_pred) < 0  # Masque des exemples de signe opposé
    doubled_loss = tf.where(mask, mse * 2, mse)  # Loss doublée pour les exemples de signe opposé
    
    return doubled_loss

traj = "C:\\Users\\PM263553\\Desktop\\finance\\Git\\DL_visionary\\resultat"


fct_3 = 'tanh'
name = "dens_3_"+fct_3

#prpose mpoi un nom de variable qui contiendra le nombre d'epoch sur lesquels on moyenne la loss pour avoir un loss moyen
n_epoch_mean = 10
learning_rate_arr = [0.03, 0.001, 0.0001, 0.00001]
activation_fct = ["relu", "tanh", "sigmoid"]
n_plot = 10
n_points = 3800
#je veux faire un dossier qui contienne les resultat d'un run total et donc qui me dise les paramètres utilisés
#pour ça je vais faire un dossier qui contient le nom de la fonction d'activation de la couche 1, le nom de la fonction d'activation de la couche 2, le learning rate et le nombre de points utilisés
traj_study = traj + "\\study_"+name +"_fct_{}_lr_{}_data_{}".format(len(activation_fct), len(learning_rate_arr), n_points)
path(traj_study)


x_length = 80
y_length = 20
largeur = x_length + y_length
scaler = StandardScaler()    



ignoreTimer = 150

ratio = 1
if n_points>2200 :
    ratio = 1.02
data = loadData(paire="BTCBUSD", sequenceLength=int(largeur*n_points*ratio), interval_str="{}m".format(5), numPartitions=4,trainProp = 0.6, validProp = 0.25, testProp  = 0.15, reload=True, ignoreTimer=ignoreTimer)
print(np.shape(data))
#%%



#on va maintenant découper ces données en segment de 80 points et mettre ça dans un array
data_cut = []
for i in range(n_points):
    part = data[i*largeur:(i+1)*largeur]
    data_cut.append(np.array(part[:]))

data_cut = np.array(data_cut)
print(np.shape(data_cut))
data_cut_flat = data_cut.reshape(data_cut.shape[0], -1)

print(np.shape(data_cut_flat))
# on va maintenant découper ces data en test train et valid en utilisant train_test_split
train, valid = train_test_split(data_cut_flat, test_size=0.15, shuffle=False)
# print(train.shape, test.shape)
# train, valid = train_test_split(train, test_size=0.25, shuffle=False)
# print(train.shape, valid.shape)

train = np.array(train.reshape(train.shape[0], largeur, 3))
#test = np.array(test.reshape(test.shape[0], largeur, 3))
valid = np.array(valid.reshape(valid.shape[0], largeur, 3))
#print(train.shape, test.shape, valid.shape)


#on veut maintenant définir les x ET les y
x_train = []
y_train = []
for i in range(len(train)):
    x_train.append(train[i][:x_length,0])
    pred_part = train[i, x_length:, 0]
    pred_output = output_pred(pred_part,train[i, x_length-1, 0])
    y_train.append(pred_output)

x_valid = []
y_valid = []
y_valid_end = []
for i in range(len(valid)):
    x_valid.append(valid[i][:x_length, 0])
    pred_part = valid[i, x_length:, 0]
    pred_output = output_pred(pred_part,valid[i, x_length-1, 0])
    y_valid.append(pred_output)
    y_valid_end.append(pred_part)
    
x_train = np.array(x_train)
x_valid = np.array(x_valid)
 # Remodeler les tableaux x_train et x_valid en 2D

# Appliquer la normalisation sur les tableaux remodelés
scaler.fit(x_train)
x_train_normalized = scaler.transform(x_train)
x_valid_normalized = scaler.transform(x_valid)

# Remodeler les tableaux normalisés en 3D
x_train_normalized_3d = x_train_normalized.reshape(x_train.shape)
x_valid_normalized_3d = x_valid_normalized.reshape(x_valid.shape)

#Given a 1-d numeric time series 𝑆 = [𝑠0, · · · , 𝑠𝑇 ] with 𝑠𝑡 ∈ R, we convert 𝑆 into a 2-d image 𝑥 by plotting it out, with 𝑡 being the horizontal axis and 𝑠𝑡 being the vertical axis1. We standardize each converted image 𝑥 through following pre-processing steps. First, pixels in 𝑥 are scaled to [0, 1] and negated (i.e., 𝑥 = 1 − 𝑥/255) so that the pixels corresponding to the plotted time series signal are bright (values close to 1), whereas the rest of the background pixels become dark (values close to 0). Note that there can be multiple bright (non-zero) pixels in each column due to anti-aliasing while plotting the images. Upon normalizing each column in 𝑥 such that the pixel values in each column sum to 1, each column can be perceived as a discrete probability distribution (see Figure 6). Columns represent the independent variable time, while rows capture the dependent variable: pixel intensity. The value of the time series 𝑆 at time 𝑡 is now simply the pixel index 𝑟 (row) at that time (column) with the highest intensity. Predictions are made over normalized data. To preserve the ability to forecast in physical units, we utilize the span of the input raw data values to transform forecasts to the corresponding physical scales.
#%%
# Définir l'entrée du modèle
loss_mean_train = np.zeros((len(activation_fct), len(activation_fct), len(learning_rate_arr)))
loss_mean_valid = np.zeros((len(activation_fct), len(activation_fct), len(learning_rate_arr)))
direction = np.zeros((len(activation_fct), len(activation_fct), len(learning_rate_arr)))
for ind_fct_1, fct_1 in enumerate(activation_fct):
    for ind_fct_2, fct_2 in enumerate(activation_fct):
        for ind_rate, learning_rate in enumerate(learning_rate_arr):
            avancement = (ind_fct_1*len(activation_fct)*len(learning_rate_arr) + ind_fct_2*len(learning_rate_arr) + ind_rate) / (len(activation_fct)*len(activation_fct)*len(learning_rate_arr)) * 100
            print("Fct_1 : {}, Fct_2 : {}, learning_rate : {}, avancement : {}%".format(fct_1, fct_2, learning_rate,int(avancement)))
            input_shape = (x_length,)
            model_choice = False
            if model_choice:
                # Taille de l'entrée (80 valeurs)
                # Définir l'encodeur
                inputs = Input(shape=input_shape)
                x = Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape)(inputs)
                x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
                x = Flatten()(x)
                encoded = Dense(units=128, activation='relu')(x)
                
                output = Dense(units=1)(encoded)
                
                encoder = Model(inputs=inputs, outputs=encoded)
                model = Model(inputs=inputs, outputs=output)
            
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mse')
            else :
                model = Sequential()
                model.add(Dense(64, activation=fct_1, input_shape=input_shape))  # Couche d'entrée
                model.add(Dense(32, activation=fct_2))  # Couche cachée
                model.add(Dense(8, activation=fct_3))  # Couche cachée
                #model.add(LSTM(units=32, activation=fct_2))
                model.add(Dense(1))  # Couche de sortie
                
            
                optimizer = Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss=custom_loss)
                #%%
            trajet = traj_study + "\\model_seq_{}_fct1_{}_fct2_{}".format(learning_rate, fct_1, fct_2)
            path(trajet)
            history = model.fit(np.array(x_train_normalized), np.array(y_train), epochs=65, batch_size=128, verbose=0)
            
            # Extraire les valeurs de la loss
            loss = history.history['loss']
            mean_loss = np.mean(loss[-n_epoch_mean:])
            loss_mean_train[ind_fct_1, ind_fct_2, ind_rate] = mean_loss
            # Tracer la courbe de la loss
            epochs = range(1, len(loss) + 1)
            plt.figure(figsize=(20, 10))
            plt.plot(epochs, loss, 'b', label='Training Loss')
            plt.title('Training Loss', fontsize=17)
            plt.xlabel('Epochs', fontsize=15)
            plt.ylabel('Loss', fontsize=15)
            plt.legend(fontsize=15)
            plt.yscale('log')
            plt.savefig(trajet+'\\loss.png', dpi=300, bbox_inches='tight')

            loss = model.evaluate(np.array(x_valid_normalized), np.array(y_valid), verbose=0)
            loss_mean_valid[ind_fct_1, ind_fct_2, ind_rate] = loss

            #on veut maintenant afficher les prédictions
            pred = model.predict(np.array(x_valid_normalized), verbose=0)
            pred_no_flat = pred
            pred = pred.reshape([-1])
            bon_dir = []
            for i in range(len(pred)):
                if pred[i]*y_valid[i]>0:
                    bon_dir.append(1.0)
                else:
                    bon_dir.append(0)
            print("Loss: {}, pourcentage de bonne direction : {}".format(np.round(loss, 5), np.round(np.sum(bon_dir)/len(bon_dir)*100),1 ))
            direction[ind_fct_1, ind_fct_2, ind_rate] = np.sum(bon_dir)/len(bon_dir)*100
            y_moy = np.mean(np.abs(y_valid))
            y_std = np.std(np.abs(y_valid))
            print("resultat moyen : {}, resultat std : {}".format(np.round(y_moy, 5), np.round(y_std, 5)))
            ecart = pred-y_valid
            ecart_moy = np.mean(np.abs(ecart))
            ecart_std = np.std(np.abs(ecart))
            print("ecart moyen : {}, ecart std : {}".format(np.round(ecart_moy, 5), np.round(ecart_std, 5)))
                
                

            for i in range(n_plot):
                plt.figure(figsize=(20, 10))
                plt.plot(valid[i, :, 0], label="close")
                end_price = valid[i, x_length-1, 0]
                plt.plot([x_length, x_length+y_length], [pred[i]*end_price+end_price, pred[i]*end_price+end_price], color = 'magenta', label="prediction")
                plt.plot([x_length, x_length+y_length], [y_valid[i]*end_price+end_price, y_valid[i]*end_price+end_price], color = 'red', label="true", linewidth=3 )
                maxi, mini = np.max(valid[i, :, 0]), np.min(valid[i, :, 0])
                plt.plot([x_length, x_length],[mini, maxi], c = "red", label = "start of prediction")
                plt.plot([x_length,  x_length+y_length],[end_price, end_price], c = "red", linestyle='dashed')
                plt.arrow(x_length, end_price, y_length, (y_valid[i])*end_price, head_width=2, head_length=2, fc='red', ec='red')
                plt.arrow(x_length, end_price, y_length, (pred[i])*end_price, head_width=2, head_length=2, fc='magenta', ec='magenta')
                plt.legend(fontsize=15)
                plt.xlabel("time (5min)", fontsize=15)
                plt.ylabel("price (USDT)", fontsize=15)
                plt.savefig(trajet+'\\pred_{}.png'.format(i), bbox_inches='tight')
                plt.close('all')
            
# on veut maintenant afficher tout ces résultats dans un graph de type heatmap
#Pour ça on va afficher les résultats sur un graph 2D 
# avec en abscisse la fonction d'activation de la couche 1 et en ordonnée la fonction d'activation de la couche 2
# dans chaque case on divisera en 4 cases avec en abscisse le learning rate et en ordonnée la loss moyenne
loss_mean_results_2D = loss_mean_train.reshape(loss_mean_train.shape[0], -1)

#%%
decim = 5
#on veut aussi une heatmap pour chaque learning rate
plt.figure(figsize=(20, 10))
for ind_rate, learning_rate in enumerate(learning_rate_arr):
    plt.clf()
    plt.imshow(loss_mean_train[:,:,ind_rate], cmap='turbo', interpolation='None')
    plt.colorbar()
    #on veut afficher la valeur de la loss dans chaque case
    for ind_fct_1, fct_1 in enumerate(activation_fct):
        for ind_fct_2, fct_2 in enumerate(activation_fct):
            text = plt.text(ind_fct_2, ind_fct_1, np.round(loss_mean_train[ind_fct_1, ind_fct_2, ind_rate], decim),
                           ha="center", va="center", color="w")
    plt.xticks(np.arange(len(activation_fct)), activation_fct)
    plt.yticks(np.arange(len(activation_fct)), activation_fct)
    plt.xlabel("activation function 2", fontsize=15)
    plt.ylabel("activation function 1", fontsize=15)
    plt.title("learning rate : {}".format(learning_rate), fontsize=15)
    save_file = traj_study + "\\heatmap_{}.png".format(learning_rate)
    plt.savefig(save_file, bbox_inches='tight')

plt.figure(figsize=(20, 10))
plt.imshow(loss_mean_results_2D, cmap='turbo', interpolation='None')
plt.colorbar()
#on veut afficher la valeur de la loss dans chaque case
for ind_fct_1, fct_1 in enumerate(activation_fct):
    for ind_fct_2, fct_2 in enumerate(activation_fct):
        for ind_rate, learning_rate in enumerate(learning_rate_arr):
            text = plt.text(ind_fct_1, ind_fct_2*len(learning_rate_arr) +  ind_rate  , np.round(loss_mean_train[ind_fct_1, ind_fct_2, ind_rate], decim),
                        ha="center", va="center", color="w")
plt.xticks(np.arange(len(learning_rate_arr)), learning_rate_arr)
plt.yticks(np.arange(len(activation_fct)), activation_fct)
plt.xlabel("learning rate", fontsize=15)
plt.ylabel("activation function", fontsize=15)
save_file = traj_study + "\\heatmap.png"
plt.savefig(save_file, bbox_inches='tight')


#%%
plt.close('all')
#on veut maintenant afficher les resultats avec un histogramme dans le style suivant : 
#je veux des couleurs qui fassent un joli dégradé sur 4 couleurs
learning_rate_colors = ["#ff0000", "#ff8000", "#ffff00", "#80ff00", "#00ff00"]
learning_rate_str = ""
for learning_rate in learning_rate_arr:
    learning_rate_str += str(learning_rate) + "_"
median = np.median(loss_mean_train)
mult = 100000
limit = median*1.75*mult
fig, ax = plt.subplots(1,1,figsize=(20, 10))
ax.set_title('Histogram of loss in TRAIN for each learning rate, activation function', fontweight ='bold', fontsize = 18)
for ind_fct_1, fct_1 in enumerate(activation_fct):
    for ind_fct_2, fct_2 in enumerate(activation_fct):
        for ind_rate, learning_rate in enumerate(learning_rate_arr):
            loss_val = loss_mean_train[ind_fct_1, ind_fct_2, ind_rate] *mult
            if loss_val > limit:
                pos_text = limit*0.9
            else:
                pos_text = loss_val*0.5
            ax.bar([ind_fct_1*len(activation_fct)*len(learning_rate_arr) + ind_fct_2*len(learning_rate_arr) + ind_rate], [loss_val], color = learning_rate_colors[ind_rate], width = 1,edgecolor ='black')
            ax.text(ind_fct_1*len(activation_fct)*len(learning_rate_arr) + ind_fct_2*len(learning_rate_arr) + ind_rate-0.5, pos_text,"{}".format(int(loss_val)),color='black', fontweight='bold',fontsize='x-large')
ax.set_ylim([0,limit])
ax.set_xticks([ind_fct_1*len(activation_fct)*len(learning_rate_arr) + ind_fct_2*len(learning_rate_arr) for ind_fct_1, fct_1 in enumerate(activation_fct) for ind_fct_2, fct_2 in enumerate(activation_fct)])
ax.set_xticklabels([fct_1 + " " + fct_2 for ind_fct_1, fct_1 in enumerate(activation_fct) for ind_fct_2, fct_2 in enumerate(activation_fct)],fontsize='large')
ax.set_yticks([i*0.0001 for i in range(6)])
ax.set_ylabel("Loss",fontweight='bold',fontsize='x-large')
ax.set_xlabel("Activation function 1, Activation function 2, Learning rate : "+learning_rate_str[:-1],fontweight='bold',fontsize='x-large')
plt.savefig(traj_study+'\\histogram_train.png', bbox_inches='tight')

#on veut la même chose mais avec la loss en valid 
median = np.median(loss_mean_valid)
mult = 100000
limit = median*1.75*mult
fig, ax = plt.subplots(1,1,figsize=(20, 10))
ax.set_title('Histogram of loss in VALID for each learning rate, activation function', fontweight ='bold', fontsize = 18)
for ind_fct_1, fct_1 in enumerate(activation_fct):
    for ind_fct_2, fct_2 in enumerate(activation_fct):
        for ind_rate, learning_rate in enumerate(learning_rate_arr):
            loss_val = loss_mean_valid[ind_fct_1, ind_fct_2, ind_rate] *mult
            if loss_val > limit:
                pos_text = limit*0.9
            else:
                pos_text = loss_val*0.5
            ax.bar([ind_fct_1*len(activation_fct)*len(learning_rate_arr) + ind_fct_2*len(learning_rate_arr) + ind_rate], [loss_val], color = learning_rate_colors[ind_rate], width = 1,edgecolor ='black')
            ax.text(ind_fct_1*len(activation_fct)*len(learning_rate_arr) + ind_fct_2*len(learning_rate_arr) + ind_rate-0.5, pos_text,"{}".format(int(loss_val)),color='black', fontweight='bold',fontsize='x-large')
ax.set_ylim([0,limit])
ax.set_xticks([ind_fct_1*len(activation_fct)*len(learning_rate_arr) + ind_fct_2*len(learning_rate_arr) for ind_fct_1, fct_1 in enumerate(activation_fct) for ind_fct_2, fct_2 in enumerate(activation_fct)])
ax.set_xticklabels([fct_1 + " " + fct_2 for ind_fct_1, fct_1 in enumerate(activation_fct) for ind_fct_2, fct_2 in enumerate(activation_fct)],fontsize='large')
ax.set_yticks([i*0.0001 for i in range(6)])
ax.set_ylabel("Loss",fontweight='bold',fontsize='x-large')
ax.set_xlabel("Activation function 1, Activation function 2, Learning rate : "+learning_rate_str[:-1],fontweight='bold',fontsize='x-large')
plt.savefig(traj_study+'\\histogram_valid.png', bbox_inches='tight')


#on veut la même chose mais avec la direction
fig, ax = plt.subplots(1,1,figsize=(20, 10))
ax.set_title('Histogram of direction for each learning rate, activation function', fontweight ='bold', fontsize = 18)
plt.plot([0, len(activation_fct)*len(activation_fct)*len(learning_rate_arr)], [50, 50], color = 'black', linestyle='dashed', linewidth=3)
for ind_fct_1, fct_1 in enumerate(activation_fct):
    for ind_fct_2, fct_2 in enumerate(activation_fct):
        for ind_rate, learning_rate in enumerate(learning_rate_arr):
            dir = direction[ind_fct_1, ind_fct_2, ind_rate]
            pos_text = dir-1
            ax.bar([ind_fct_1*len(activation_fct)*len(learning_rate_arr) + ind_fct_2*len(learning_rate_arr) + ind_rate], [dir], color = learning_rate_colors[ind_rate], width = 1,edgecolor ='black')
            ax.text(ind_fct_1*len(activation_fct)*len(learning_rate_arr) + ind_fct_2*len(learning_rate_arr) + ind_rate-0.5, pos_text,"{}".format(int(dir)),color='black', fontweight='bold',fontsize='x-large')
ax.set_ylim([np.min(direction)-3,np.max(direction)+3])
ax.set_xticks([ind_fct_1*len(activation_fct)*len(learning_rate_arr) + ind_fct_2*len(learning_rate_arr) for ind_fct_1, fct_1 in enumerate(activation_fct) for ind_fct_2, fct_2 in enumerate(activation_fct)])
ax.set_xticklabels([fct_1 + " " + fct_2 for ind_fct_1, fct_1 in enumerate(activation_fct) for ind_fct_2, fct_2 in enumerate(activation_fct)],fontsize='large')
ax.set_ylabel("Direction",fontweight='bold',fontsize='x-large')
ax.set_xlabel("Activation function 1, Activation function 2, Learning rate : "+learning_rate_str[:-1],fontweight='bold',fontsize='x-large')
plt.savefig(traj_study+'\\histogram_dir.png', bbox_inches='tight')
