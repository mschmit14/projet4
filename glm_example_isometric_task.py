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
file_name = "Hugo_friction_block5.txt"
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

absLFv = np.abs(LFv) # Absolute value of LFv
meanLFv = np.mean(absLFv) # Mean value of LFv
meanLFv = np.ones(len(LFv))*meanLFv # Obtain a vector of the same length as LFv with the mean value of LFv

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
#%% Friction coefficient
#Index
mask_R = np.abs(Fy_R) < 0.2
mu_R = np.where(mask_R, np.nan, np.sqrt((np.square(Fx_R))+np.square(Fy_R))/np.sqrt((np.square(Fy_R)))) # Friction coefficient from GF to LF
print('Friction coefficient for the index : ' + str(np.nanmean(mu_R)))

#Thumb
mask_L = np.abs(Fy_L) < 0.2
mu_L = np.where(mask_L, np.nan, np.sqrt((np.square(Fx_L)+np.square(Fy_L)))/np.sqrt((np.square(Fy_L)))) # Friction coefficient from GF to LF
print('Friction coefficient for the thumb : ' + str(np.nanmean(mu_L)))

#%% COP peaks 
from scipy.signal import find_peaks

#Thumb
CPx_L_filtered = CPx_L.copy()
CPx_L_filtered[CPx_L_filtered > 0.02] = np.nan
CPx_L_filtered[CPx_L_filtered < -0.02] = np.nan
peaks_L, _ = find_peaks(CPx_L_filtered, distance=1000, height=0.003)  
npeaks_L, _ = find_peaks(-CPx_L_filtered, distance=1000, height=0.003)  

#Index
CPx_R_filtered = CPx_R.copy()
CPx_R_filtered[CPx_R_filtered > 0.02] = np.nan
CPx_R_filtered[CPx_R_filtered < -0.02] = np.nan
peaks_R, _ = find_peaks(CPx_R_filtered, distance=1000, prominence=0.01, height=0.003)  
npeaks_R, _ = find_peaks(-CPx_R_filtered, distance=1000, prominence=0.01, height=0.003)  

#%% useful zone
upper_threshold = 0.003
lower_threshold = -0.003

from scipy.signal import find_peaks

# Thumb
# Find the indices of the useful zones
useful_zone_indices_upper_L = np.where(CPx_L_filtered < upper_threshold)[0]
useful_zone_indices_lower_L = np.where(CPx_L_filtered > lower_threshold)[0]

# Create subsets of mu_L for each useful zone
mu_L_upper_zone = mu_L[useful_zone_indices_upper_L]
mu_L_lower_zone = mu_L[useful_zone_indices_lower_L]

# Find the peaks in each useful zone
peaks_upper_zone_L, _ = find_peaks(mu_L_upper_zone, distance=500, prominence=0.01)
peaks_lower_zone_L, _ = find_peaks(mu_L_lower_zone, distance=500, prominence=0.01)

# Convert the peak indices back to the original indices
peaks_upper_zone_L = useful_zone_indices_upper_L[peaks_upper_zone_L]
peaks_lower_zone_L = useful_zone_indices_lower_L[peaks_lower_zone_L]


# Index
# Find the indices of the useful zones
useful_zone_indices_upper_R = np.where(CPx_R_filtered < upper_threshold)[0]
useful_zone_indices_lower_R = np.where(CPx_R_filtered > lower_threshold)[0]

# Create subsets of mu_R for each useful zone
mu_R_upper_zone = mu_R[useful_zone_indices_upper_R]
mu_R_lower_zone = mu_R[useful_zone_indices_lower_R]

# Find the peaks in each useful zone
peaks_upper_zone_R, _ = find_peaks(mu_R_upper_zone, distance=500, prominence=0.01)
peaks_lower_zone_R, _ = find_peaks(mu_R_lower_zone, distance=500, prominence=0.01)

# Convert the peak indices back to the original indices
peaks_upper_zone_R = useful_zone_indices_upper_R[peaks_upper_zone_R]
peaks_lower_zone_R = useful_zone_indices_lower_R[peaks_lower_zone_R]

# Combine peaks_upper_zone and peaks_lower_zone into a single array
sum_peaks_L = np.concatenate((peaks_upper_zone_L, peaks_lower_zone_L))

# Combine peaks_upper_zone and peaks_lower_zone into a single array
sum_peaks_R = np.concatenate((peaks_upper_zone_R, peaks_lower_zone_R))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))

# Plot the mu_R friction coefficient on the first subplot
ax1.plot(time, mu_R, label='mu_R')
ax1.set_ylabel('mu_R')
ax1.plot(peaks_upper_zone_R/1000, mu_R[peaks_upper_zone_R], "x", label='Peaks in upper zone')
ax1.plot(peaks_lower_zone_R/1000, mu_R[peaks_lower_zone_R], "x", label='Peaks in lower zone')
ax1.set_xlim([0, 35])  # Limit the x-axis between 0 and 35
ax1.legend()

# Plot the mu_L friction coefficient on the second subplot
ax2.plot(time, mu_L, label='mu_L')
ax2.plot(peaks_upper_zone_L/1000, mu_L[peaks_upper_zone_L], "x", label='Peaks in upper zone')
ax2.plot(peaks_lower_zone_L/1000, mu_L[peaks_lower_zone_L], "x", label='Peaks in lower zone')
ax2.set_ylabel('mu_L')
ax2.set_xlim([0, 35])  # Limit the x-axis between 0 and 35
ax2.legend()

# Plot the CPx_L data and the peaks on the third subplot
ax3.plot(np.arange(len(CPx_L_filtered))/1000, CPx_L_filtered, label='CPx_L')
ax3.plot(peaks_L/1000, CPx_L_filtered[peaks_L], "x", label='Positive peaks')
ax3.plot(npeaks_L/1000, CPx_L_filtered[npeaks_L], "x", label='Negative peaks')
ax3.set_xlim([0, 35])  # Limit the x-axis between 0 and 35
ax3.set_ylim([-0.03, 0.03])  # Limit the y-axis between -0.02 and 0.02
ax3.axhline(y=upper_threshold, color='r', linestyle='--', label='Upper Threshold')  # Add the upper threshold line
ax3.axhline(y=lower_threshold, color='b', linestyle='--', label='Lower Threshold')  # Add the lower threshold line
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('CPx_L')
ax3.legend()

# Plot the CPx_R data and the peaks on the fourth subplot
ax4.plot(np.arange(len(CPx_R_filtered))/1000, CPx_R_filtered, label='CPx_R')
ax4.plot(peaks_R/1000, CPx_R_filtered[peaks_R], "x", label='Positive peaks')
ax4.plot(npeaks_R/1000, CPx_R_filtered[npeaks_R], "x", label='Negative peaks')
ax4.set_xlim([0, 35])  # Limit the x-axis between 0 and 35
ax4.set_ylim([-0.03, 0.03])  # Limit the y-axis between -0.02 and 0.02
ax4.axhline(y=upper_threshold, color='r', linestyle='--', label='Upper Threshold')  # Add the upper threshold line
ax4.axhline(y=lower_threshold, color='b', linestyle='--', label='Lower Threshold')  # Add the lower threshold line
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('CPx_R')
ax4.legend()

plt.tight_layout()
plt.show()
plt.close()

#%% CF
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Create a new figure for the Fx_L and Fx_R vs peaks plot
fig, axs = plt.subplots(2, figsize=(10, 16))

# For Left
# Give the y of every peak
CF_L = mu_L[sum_peaks_L]
NF_L = np.sqrt((np.square(Fx_L[sum_peaks_L])))

# Replace NaN values in CF with the mean of the non-NaN elements
CF_L = np.where(np.isnan(CF_L), np.nanmean(CF_L), CF_L)

# Convert CF and NF to logarithmic scale
log_CF_L = np.log(CF_L)
log_NF_L = np.log(NF_L)

# Reshape log_NF to be a 2D array
log_NF_L = log_NF_L.reshape(-1, 1)

# Fit the linear regression model
model_L = LinearRegression().fit(log_NF_L, log_CF_L)

# The coefficient 'n' is the slope of the regression line
n_L = model_L.coef_[0]

# The intercept 'log(k)' is the y-intercept of the regression line
log_k_L = model_L.intercept_

# Convert 'log(k)' back to 'k'
k_L = np.exp(log_k_L)

print(f"Left: k: {k_L}, n: {n_L}")

# Plot Fx_L values at the peaks in the upper zone
axs[0].scatter(NF_L, CF_L, marker="x", label='Actual CF')

# Calculate CF using the formula
calculated_CF_L = k_L * (np.linspace(0.1, 10) ** n_L)

# Plot the calculated CF values
axs[0].plot(np.linspace(0.1, 10), calculated_CF_L, color='red', label='Calculated CF')

axs[0].set_xlabel('NF')
axs[0].set_ylabel('CF')
axs[0].legend()
axs[0].set_title('Left')

# For Right
# Give the y of every peak
CF_R = mu_R[sum_peaks_R]
NF_R = np.sqrt((np.square(Fx_R[sum_peaks_R])))

# Replace NaN values in CF with the mean of the non-NaN elements
CF_R = np.where(np.isnan(CF_R), np.nanmean(CF_R), CF_R)

# Convert CF and NF to logarithmic scale
log_CF_R = np.log(CF_R)
log_NF_R = np.log(NF_R)

# Reshape log_NF to be a 2D array
log_NF_R = log_NF_R.reshape(-1, 1)

# Fit the linear regression model
model_R = LinearRegression().fit(log_NF_R, log_CF_R)

# The coefficient 'n' is the slope of the regression line
n_R = model_R.coef_[0]

# The intercept 'log(k)' is the y-intercept of the regression line
log_k_R = model_R.intercept_

# Convert 'log(k)' back to 'k'
k_R = np.exp(log_k_R)

print(f"Right: k: {k_R}, n: {n_R}")

# Plot Fx_R values at the peaks in the upper zone
axs[1].scatter(NF_R, CF_R, marker="x", label='Actual CF')

# Calculate CF using the formula
calculated_CF_R = k_R * (np.linspace(0.1, 10) ** n_R)

# Plot the calculated CF values
axs[1].plot(np.linspace(0.1, 10), calculated_CF_R, color='red', label='Calculated CF')

axs[1].set_xlabel('NF')
axs[1].set_ylabel('CF')
axs[1].legend()
axs[1].set_title('Right')

plt.tight_layout()
plt.show()
plt.close()

#%% Export data
import os 


#%% Moyenne glissante
window_size = 3 * freqAcq #3000
window = np.ones(window_size)/window_size
LFv_smooth = np.convolve(absLFv, window, 'same')

#%%
# Trouver les indices des points d'intersection entre LFv_smooth et meanLFv
intersection_indices = np.where(np.diff(np.sign(LFv_smooth - absLFv)))[0]

# Définir la taille de la fenêtre pour la moyenne glissante autour des points d'intersection
window_size_intersection = 100  # Vous devrez ajuster cela en fonction de vos besoins

# Initialiser une liste pour stocker les valeurs de moyenne glissante autour des points d'intersection
smoothed_intersection_values = []

# Calculer la moyenne glissante autour de chaque point d'intersection
for index in intersection_indices:
    start_index = max(0, index - window_size_intersection // 2)
    end_index = min(len(LFv_smooth), index + window_size_intersection // 2)
    smoothed_value = np.mean(LFv_smooth[start_index:end_index])
    smoothed_intersection_values.append(smoothed_value)
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Create a new figure for the Fx vs NF plot
plt.figure(figsize=(10, 8))

# Plot Fx_L values at the peaks in the upper zone for each intensity
for intensity, color in zip([0, 1, 2], ['blue', 'green', 'red']):
    CF = mu_L[sum_peaks_L[intensity]]
    NF = np.sqrt((np.square(Fx_L[sum_peaks_L[intensity]])))
    CF = np.where(np.isnan(CF), np.nanmean(CF), CF)
    log_CF = np.log(CF)
    log_NF = np.log(NF)
    log_NF = log_NF.reshape(-1, 1)
    model = LinearRegression().fit(log_NF, log_CF)
    n = model.coef_[0]
    log_k = model.intercept_
    k = np.exp(log_k)
    calculated_CF = k * (np.linspace(0.1, 10) ** n)
    plt.scatter(NF, CF, marker="x", label=f'Actual CF - Intensity {intensity}', color=color)
    plt.plot(np.linspace(0.1, 10), calculated_CF, color=color, label=f'Calculated CF - Intensity {intensity}')

plt.xlabel('NF')
plt.ylabel('CF')
plt.legend()
plt.title('Combined CF Plot for Different Intensities (Left)')
plt.show()
plt.close()


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
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
ax[3].axhline(y=0, color='g', linestyle='--', linewidth=1)
ax[3].set_ylabel("COP [mm]", fontsize=12)
ax[3].legend(fontsize=12)
ax[3].set_ylim([-20,20])
ax[3].set_xlabel("Time [s]", fontsize=13)



plt.show()
        

        
    
    
