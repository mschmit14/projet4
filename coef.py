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

#%% CF
from sklearn.linear_model import LinearRegression

        
#%% Import data

# File path
data_dir = "./YOUR/DATA/DIRECTORY"
#file_name = "Hugo_friction_gant_final_block32.txt"

victor_sans_debut = ["Victor_friction_block61.txt","Victor_friction_block62.txt","Victor_friction_block63.txt"]
victor_sans_fin = ["Victor_friction_final_block84.txt","Victor_friction_final_block85.txt","Victor_friction_final_block86.txt"]
victor_avec_debut = ["Victor_friction_gant_block64.txt","Victor_friction_gant_block65.txt","Victor_friction_gant_block66.txt"]
victor_avec_fin = ["Victor_friction_gant_final_block88.txt","Victor_friction_gant_final_block89.txt","Victor_friction_gant_final_block90.txt"]

lise_sans_debut = ["Lise_friction_block33.txt","Lise_friction_block34.txt","Lise_friction_block35.txt"]
lise_sans_fin = ["Lise_friction_final_block55.txt","Lise_friction_final_block56.txt","Lise_friction_final_block57.txt"]
lise_avec_debut = ["Lise_friction_gant_block36.txt","Lise_friction_gant_block37.txt","Lise_friction_gant_block38.txt"]
lise_avec_fin = ["Lise_friction_gant_final_block58.txt","Lise_friction_gant_final_block59.txt","Lise_friction_gant_final_block60.txt"]

sophie_sans_debut = ["Sophie_friction_block91.txt","Sophie_friction_block92.txt","Sophie_friction_block93.txt"]
sophie_sans_fin = ["Sophie_friction_final_block113.txt","Sophie_friction_final_block114.txt","Sophie_friction_final_block115.txt"]
sophie_avec_debut = ["Sophie_friction_gant_block94.txt","Sophie_friction_gant_block95.txt","Sophie_friction_gant_block96.txt"]
sophie_avec_fin = ["Sophie_friction_gant_final_block116.txt","Sophie_friction_gant_final_block117.txt","Sophie_friction_gant_final_block118.txt"]

hugo_sans_debut = ["Hugo_friction_block5.txt","Hugo_friction_block6.txt","Hugo_friction_block7.txt"]
hugo_sans_fin = ["Hugo_friction_final_block27.txt","Hugo_friction_final_block28.txt","Hugo_friction_final_block29.txt"]
hugo_avec_debut = ["Hugo_friction_gant_block8.txt","Hugo_friction_gant_block9.txt","Hugo_friction_gant_block10.txt"]
hugo_avec_fin = ["Hugo_friction_gant_final_block30.txt","Hugo_friction_gant_final_block31.txt","Hugo_friction_gant_final_block32.txt"]


liste = victor_sans_debut


log_CF_L_tot = []
log_NF_L_tot = []
log_CF_R_tot = []
log_NF_R_tot = []

for name in liste : 
    file_name = name 
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
    # sum = 0
    # count = 0
    # for i in range(len(GF)):
    #     if GF[i] < 0.1:
    #         sum+=1
    #         # print("GF < 0.1 : " + str(GF[i]))
    # GF_new = np.ndarray(len(GF)-sum)
    # print("sum = " + str(sum)) #4376 ; 1450 ; 2998
    # for i in range(len(GF)):
    #     if GF[i] < 0.1:
    #         sum+=1
    #         # print("GF < 0.1 : " + str(GF[i]))
    #     else :
    #         GF_new[count] = GF[i]
    #         count+=1
    # print(len(GF_new))
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


    #%% Friction coefficient
    #Index
    mask_R = np.abs(Fy_R) < 0.2
    # print("len de fy_R" +str(len(Fy_R)))

    sum = 0
    # print(Fy_R)
    for i in range(len(Fy_R)):
        if abs(Fy_R[i]) < 0.05:
            sum+=1
            # print("Fy_R < 0.1 : " + str(Fy_R[i]))
    # print("sum = " + str(sum)) #4376 ; 1450 ; 2998
    new_Fy_R= np.ndarray(len(Fy_R-sum))

    cnt = 0
    for i in range(len(Fy_R)):
        if abs(Fy_R[i]) > 0.05:
            new_Fy_R[cnt] = Fy_R[i]
            cnt+=1

    sum2 = 0
    # print(Fy_L)
    for i in range(len(Fy_L)):
        if abs(Fy_L[i]) < 0.04:
            sum2+=1
            # print("Fy_R < 0.1 : " + str(Fy_R[i]))
    # print("sum2 = " + str(sum2)) #4376 ; 1450 ; 2998
    new_Fy_L= np.ndarray(len(Fy_L-sum2))

    cnt2 = 0
    for i in range(len(Fy_L)):
        if abs(Fy_L[i]) > 0.04:
            new_Fy_L[cnt2] = Fy_L[i]
            cnt2+=1

    mu_R = np.where(mask_R, np.nan, np.sqrt((np.square(Fx_R))+np.square(Fz_R))/np.sqrt((np.square(new_Fy_R)))) # Friction coefficient from GF to LF
    print('Friction coefficient for the index : ' + str(np.nanmean(mu_R)))

    #Thumb
    mask_L = np.abs(Fy_L) < 0.2
    mu_L = np.where(mask_L, np.nan, np.sqrt((np.square(Fx_L)+np.square(Fz_L)))/np.sqrt((np.square(new_Fy_L)))) # Friction coefficient from GF to LF
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

   

    # For Left
    # Give the y of every peak
    CF_L = mu_L[sum_peaks_L]
    NF_L = np.sqrt((np.square(Fx_L[sum_peaks_L])))

    # Replace NaN values in CF with the mean of the non-NaN elements
    CF_L = np.where(np.isnan(CF_L), np.nanmean(CF_L), CF_L)

    # Convert CF and NF to logarithmic scale
    log_CF_L = np.log(CF_L)
    log_NF_L = np.log(NF_L)

    log_CF_L_tot += list(log_CF_L)
    log_NF_L_tot += list(log_NF_L)

    # For Right
    # Give the y of every peak
    CF_R = mu_R[sum_peaks_R]
    NF_R = np.sqrt((np.square(Fx_R[sum_peaks_R])))

    # Replace NaN values in CF with the mean of the non-NaN elements
    CF_R = np.where(np.isnan(CF_R), np.nanmean(CF_R), CF_R)

    # Convert CF and NF to logarithmic scale
    log_CF_R = np.log(CF_R)
    log_NF_R = np.log(NF_R)

    log_CF_R_tot += list(log_CF_R)	
    log_NF_R_tot += list(log_NF_R)


#%% CF

# Create a new figure for the Fx_L and Fx_R vs peaks plot
fig, axs = plt.subplots(2, figsize=(6, 6))


# Reshape log_NF to be a 2D array
NF_L_tot = np.exp(log_NF_L_tot)
CF_L_tot = np.exp(log_CF_L_tot)
NF_R_tot = np.exp(log_NF_R_tot)
CF_R_tot = np.exp(log_CF_R_tot)


log_NF_L_tot = np.array(log_NF_L_tot).reshape(-1, 1)
log_CF_L_tot  = np.array(log_CF_L_tot)
log_NF_R_tot = np.array(log_NF_R_tot).reshape(-1, 1)
log_CF_R_tot = np.array(log_CF_R_tot)


# Fit the linear regression model
model_L = LinearRegression().fit(log_NF_L_tot, log_CF_L_tot)

# The coefficient 'n' is the slope of the regression line
n_L = model_L.coef_[0]

# The intercept 'log(k)' is the y-intercept of the regression line
log_k_L = model_L.intercept_

# Convert 'log(k)' back to 'k'
k_L = np.exp(log_k_L)

print(f"Left: k: {k_L}, n: {n_L}")

# Plot Fx_L values at the peaks in the upper zone
axs[0].scatter(NF_L_tot, CF_L_tot, marker="x", label='Actual CF')

# Calculate CF using the formula
calculated_CF_L = k_L * (np.linspace(0.1, 10) ** n_L)

# Plot the calculated CF values
axs[0].plot(np.linspace(0.1, 10), calculated_CF_L, color='red', label='Calculated CF')

axs[0].set_xlabel('NF')
axs[0].set_ylabel('CF')
axs[0].legend()
axs[0].set_title('Left')



# Fit the linear regression model
model_R = LinearRegression().fit(log_NF_R_tot, log_CF_R_tot)

# The coefficient 'n' is the slope of the regression line
n_R = model_R.coef_[0]

# The intercept 'log(k)' is the y-intercept of the regression line
log_k_R = model_R.intercept_

# Convert 'log(k)' back to 'k'
k_R = np.exp(log_k_R)

print(f"Right: k: {k_R}, n: {n_R}")

# Plot Fx_R values at the peaks in the upper zone
axs[1].scatter(NF_R_tot, CF_R_tot, marker="x", label='Actual CF')

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
LFv_smooth = np.convolve(np.abs(LFv), window, 'same')

#%%
# Trouver les indices des points d'intersection entre LFv_smooth et meanLFv
intersection_indices = np.where(np.diff(np.sign(LFv_smooth - np.abs(LFv))))[0]

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