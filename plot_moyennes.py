import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import make_interp_spline, BSpline
#from IPython.display import display, Markdown as md


# Import custom functions
import signal_processing_toolbox as processing

def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name
        
file_name = "moyenne.txt"
data = "data.csv"

ligne1 = []

down_tot = []
up_tot = []


with open(file_name, 'r') as fichier:
    lignes = fichier.readlines()

df = pd.read_csv(data, sep=';')
# display(df)

#ajouter la colonne LF
df['LF'] = 0
df['GF'] = 0
df['LF'] = df['LF'].apply(lambda x: [] if pd.isna(x) else x)
df['GF'] = df['GF'].apply(lambda x: [] if pd.isna(x) else x)

for i in range(16):
    LF_init = lignes[2*i].strip().split()
    GF_init = lignes[2*i+1].strip().split()

    print(len(LF_init))

    LF_post = [0]*len(LF_init)
    print("LF_post : ", LF_post)
    print("LF_init : ", LF_init)
    print("LF_post size : ", len(LF_post))
    print("LF_init size : ", len(LF_init))
    GF_post = [0]*len(GF_init)
    print("GF_post : ", GF_post)
    print("GF_init : ", GF_init)
    print("GF_post size : ", len(GF_post))
    print("GF_init size : ", len(GF_init))

    for j in range (len(LF_init)):
        print("i'm here \n")
        LF_post[j] = float(LF_init[j])
        GF_post[j] = float(GF_init[j])

    print("LF_post : ", LF_post)
    print("LF_post size : ", len(LF_post))
    print("GF_post : ", GF_post)
    print("GF_post size : ", len(GF_post))

    df.loc[i, 'LF'] = LF_post
    df.loc[i, 'GF'] = GF_post

# display(df)
# print("LF : ", df['LF'][0])
# print("GF : ", df['GF'][0])

Victor_LF = [0,2,4,6,8,10,12,14]
Victor_GF= [1,3,5,7,9,11,13,15]

Victor_entr_LF = [16,18,20,22,24,26,28,30]
Victor_entr_GF = [17,19,21,23,25,27,29,31]

Lise_LF = [32,34,36,38,40,42,44,46]
Lise_GF = [33,35,37,39,41,43,45,47]

Lise_entr_LF = [48,50,52,54,56,58,60,62]
Lise_entr_GF = [49,51,53,55,57,59,61,63]

Hugo_LF = [64,66,68,70,72,74,76,78]
Hugo_GF = [65,67,69,71,73,75,77,79]

Hugo_entr_LF = [80,82,84,86,88,90,92,94]
Hugo_entr_GF = [81,83,85,87,89,91,93,95]

Sophie_LF = [96,98,100,102,104,106,108,110]
Sophie_GF = [97,99,101,103,105,107,109,111]

Sophie_entr_LF = [112,114,116,118,120,122,124,126]
Sophie_entr_GF = [113,115,117,119,121,123,125,127]

liste_utilisee = Hugo_LF

fig = plt.figure(figsize = [15,9])

for j in liste_utilisee:
    ligne1 = lignes[j].strip().split()

    for i in range (len(ligne1)):
        ligne1[i] = float(ligne1[i])

    print("Liste 1:", ligne1)

    #create a new list for the even indexes of the line
    down = []
    for i in range(0, len(ligne1), 2):
        down.append(ligne1[i])

    #create a new list for the odd indexes of the line
    up = []
    for i in range(1, len(ligne1), 2):
        up.append(ligne1[i])

    print("down:", down)

    for i in range (len(down)):
        down_tot.append(down[i])
    for i in range (len(up)):
        up_tot.append(up[i])

    # x_axis is the number of elements in a line
    x_axis_down = np.arange(0, len(ligne1), 2) 
    x_axis_up = np.arange(1, len(ligne1), 2)

    plt.xticks(np.arange(0, len(ligne1)))


    # plt.scatter(x_axis_down, down, label = "Down")
    # plt.scatter(x_axis_up, up, label = "Up")

    # xnew_down = np.linspace(x_axis_down.min(), x_axis_down.max(), 500) 
    # xnew = np.linspace(x_axis_up.min(), x_axis_up.max(), 500)

    # spl = make_interp_spline(x_axis_down, down, k=3)  # type: BSpline
    # down_smooth = spl(xnew_down)
    # plt.plot(xnew_down, down_smooth)

    # spl = make_interp_spline(x_axis_up, up, k=3)  # type: BSpline
    # up_smooth = spl(xnew)
    # plt.plot(xnew, up_smooth)

xs_down = []
for i in range(len(down_tot)):
    xs_down.append(np.random.normal((i+1)*10e-20,0.02))

xs_up = []
for i in range(len(up_tot)):
    xs_up.append(np.random.normal((i+1)*10e-20 +1,0.02))
#create box plots for the even indexes of the line
plt.boxplot(down_tot, positions=[0],labels=['down'])
for xs,val in zip(xs_down, down_tot):
    plt.scatter(xs, val, color='red')
#create box plots for the odd indexes of the line
plt.boxplot(up_tot, positions=[1],labels=['up'])
for xs,val in zip(xs_up, up_tot):
    plt.scatter(xs, val, color='green')

plt.grid()
plt.title(get_var_name(liste_utilisee))
plt.show()