# -*- coding: utf-8 -*-
"""
Example script for extracting, processing and plotting data acquired with the 
PFC 58 LabView VI.

Created on Fri Feb 24 2023

@author: opsomerl

"""
#%% Importation des librairies necessaires
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Import custom functions
import signal_processing_toolbox as processing

        
#%% Import data

# File path
data_dir = "./YOUR/DATA/DIRECTORY"
file_name = "Lise_block50.txt"

# Import data from txt file
#df = pd.read_csv(data_dir + file_name, sep = '\t', header = None)
df = pd.read_csv(file_name, 
                 sep = '\t', 
                 header = None)


#%% Extract force data from data frame. In the example data provided, data from two 
# different setups were acquired in parallel (two subjects performed the experiment 
# simultaneously). In your case, you will acquire data from only one manipulandum.

# Time
freqAcq=1000 #Acquisition sampling frequency (Hz)                 
n = len(df.iloc[:,0])       
time = np.linspace(0,(n-1)/freqAcq,n)

# Forces and torques from ATI sensors
F_L = -df.iloc[:,0:3].to_numpy()  # Left ATI (N)
T_L = -df.iloc[:,3:6].to_numpy()  # Left ATI (Nm)
F_R = -df.iloc[:,6:9].to_numpy()  # Right ATI (N)
T_R = -df.iloc[:,9:12].to_numpy() # Right ATI (Nm)

# Accelerometer
acc = df.iloc[:,12:15].to_numpy()

# Metronome
audio_fb = df.iloc[0:-1:20,15].to_numpy()
time_fb = time[0:-1:20]

# LEDs
LED_up = df.iloc[0:-1:20,16].to_numpy()
LED_down = df.iloc[1:-1:20,16].to_numpy()

# Humidity on left sensor (index finger)
mev_L = df.iloc[:,20]


#%% Align force/torque data with manipulandum ref frame

# The axes of the ATI sensors are not aligned with the axes of the manipulandum. We need to 
# rotate the forces to align them with the vertical and horizontal axes
alpha = np.radians(30) # rotation angle of the ATI relative to the manipulandum

# Forces
Fx_R = -F_R[:,0]*np.sin(alpha) - F_R[:,1]*np.cos(alpha) # vertical tangential component on right sensor
Fy_R = -F_R[:,2]                                        # normal component on right sensor
Fz_R = -F_R[:,0]*np.cos(alpha) + F_R[:,1]*np.sin(alpha) # horizontal tangential component on left sensor

Fx_L =  F_L[:,0]*np.sin(alpha) + F_L[:,1]*np.cos(alpha) # vertical tangential component on left sensor
Fy_L =  F_L[:,2]                                        # normal component on left sensor
Fz_L = -F_L[:,0]*np.cos(alpha) + F_L[:,1]*np.sin(alpha) # horizontal tangential component on left sensor

# Torques
Tx_R = -T_R[:,0]*np.sin(alpha) - T_R[:,1]*np.cos(alpha) 
Ty_R = -T_R[:,2]                                        
Tz_R = -T_R[:,0]*np.cos(alpha) + T_R[:,1]*np.sin(alpha) 

Tx_L =  T_L[:,0]*np.sin(alpha) + T_L[:,1]*np.cos(alpha) 
Ty_L =  T_L[:,2]                                        
Tz_L = -T_L[:,0]*np.cos(alpha) + T_L[:,1]*np.sin(alpha) 

# Load Force (LF) and Grip Force (GF)
GF  = 0.5*(abs(Fy_L) + abs(Fy_R))  # GF is defined as the average of the left and right normal forces
LFv = Fx_L + Fx_R                  # Vertical component of LF
LFh = Fz_L + Fz_R                  # Horizontal component of LF
LF  = np.hypot(LFv,LFh)            # LF norm


#%% Compute centers of pressure = point of application of resultant force
z0 = 1.55e-3 # Distance between GLM origin and contact surface [m]

CPx_L =  (Tz_L - Fx_L*z0)/Fy_L
CPz_L = -(Tx_L + Fz_L*z0)/Fy_L  

CPx_R =  (Tz_R + Fx_R*z0)/Fy_R
CPz_R = -(Tx_R - Fz_R*z0)/Fy_R 
       

#%% Filter data
freqFiltForces=20 #Low-pass filter cut-off frequency for force signals (Hz)

GF  = processing.filter_signal(GF,   fs = freqAcq, fc = freqFiltForces)
LF  = processing.filter_signal(LF,   fs = freqAcq, fc = freqFiltForces)
LFv = processing.filter_signal(LFv,  fs = freqAcq, fc = freqFiltForces)
LFh = processing.filter_signal(LFh,  fs = freqAcq, fc = freqFiltForces)


#%% LF derivative
dLFv = processing.derive(LFv, 1000)


#%% Basic plot of the data

# Close figures
plt.close('all')

# Initialize new figure
fig = plt.figure(figsize = [15,9])
ax  = fig.subplots(4, 1, sharex=True)

# LEDs
ax[0].plot(time_fb, LED_up, label='UP')
ax[0].plot(time_fb, LED_down, label='DOWN')
ax[0].set_ylim([-0.5,1.5])
ax[0].set_ylabel("LED", fontsize=12)
ax[0].legend(fontsize=12)
ax[0].set_title("Simple example of isometric force data", fontsize=14, fontweight="bold")
ax[0].set_yticks([0,1])
ax[0].set_yticklabels(['OFF','ON'])


# LF
ax[1].plot([time[0],time[-1]], [0, 0], color=[0.6,0.6,0.6], linewidth=0.5)
ax[1].plot(time, np.abs(LFv))
ax[1].set_ylabel("LFv [N]", fontsize=12)
#défini la graduation de l'axe des ordonnées
ax[1].yaxis.set_major_locator(MultipleLocator(2.5))

#ajoute une ligne horizontale à 5N et -5N sur le graphe pour mieux visualiser
ax[1].axhline(y=5, color='r', linestyle='--', linewidth=1)
ax[1].axhline(y=-5, color='r', linestyle='--', linewidth=1)

#calcul de la moyenne de la valeur absolue de LF
mean_LF = np.mean(np.abs(LFv))
ax[1].axhline(y = mean_LF, color='b', linewidth=1)

#calcul moyenne glissante
data_pd= pd.DataFrame(np.abs(LFv))
LFv_ma = data_pd.rolling(window=6000, min_periods=1).mean()  # Ensure minimum period is set to 1
LFv_ma_values = LFv_ma.iloc[:, 0].values  # Extracting values from the DataFrame
ax[1].plot(time, LFv_ma_values)

#calcul des points d'intersection entre la moyenne et LF
diff_lf = np.abs(LFv) -mean_LF
crossings_indices_lf = np.where(np.diff(np.sign(diff_lf)))[0]
cross_times_lf = time[crossings_indices_lf]


#calcul de la moyenne au milieu du plateau de lf
moy_plateau_lf = []
for i in range(len(crossings_indices_lf)//2):
    inter = crossings_indices_lf[2*i+1]-crossings_indices_lf[2*i]
    quart = inter//4
    mean = np.mean(np.abs(LFv)[crossings_indices_lf[2*i]+quart:crossings_indices_lf[2*i+1]-quart])
    moy_plateau_lf.append(mean)




# GF
ax[2].plot([time[0],time[-1]], [0, 0], color=[0.6,0.6,0.6], linewidth=0.5)
ax[2].plot(time, GF)
ax[2].set_ylabel("GF [N]", fontsize=12)
#défini la graduation de l'axe des ordonnées
ax[2].yaxis.set_major_locator(MultipleLocator(2.5))

#ajoute une ligne horizontale à 5N et 10N sur le graphe pour mieux visualiser
ax[2].axhline(y=5, color='g', linestyle='--', linewidth=1)
ax[2].axhline(y=10, color='b', linestyle='--', linewidth=1)

#calcul de la moyenne de GF
mean_GF = np.mean(GF)
ax[2].axhline(y = mean_GF, color='r', linewidth=1)

#calcul moyenne glissante
data_pd_gf= pd.DataFrame(GF)
GF_ma = data_pd_gf.rolling(window=6000, min_periods=1).mean()  # Ensure minimum period is set to 1
GF_ma_values = GF_ma.iloc[:, 0].values  # Extracting values from the DataFrame
ax[2].plot(time, GF_ma_values)

#calcul des points d'intersection entre la moyenne et GF
diff_gf = GF - mean_GF
crossings_indices_gf = np.where(np.diff(np.sign(diff_gf)))[0]
cross_times_gf = time[crossings_indices_gf]
print(len(crossings_indices_gf))
print(cross_times_gf)

#calcul de la moyenne au milieu du plateau de gf
moy_plateau_gf= []
for i in range(len(crossings_indices_gf)//2):
    inter = crossings_indices_gf[2*i+1]-crossings_indices_gf[2*i]
    quart = inter//4
    mean = np.mean(GF[crossings_indices_gf[2*i]+quart:crossings_indices_gf[2*i+1]-quart])
    moy_plateau_gf.append(mean)
print(moy_plateau_gf)


# Centers of pressure
ax[3].plot(time, CPx_L*1000, label="Index")
ax[3].plot(time, CPx_R*1000, label="Thumb")
ax[3].set_ylabel("COP [mm]", fontsize=12)
ax[3].legend(fontsize=12)
ax[3].set_ylim([-20,20])
ax[3].set_xlabel("Time [s]", fontsize=13)

plt.show()
        

        
    
    
