import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM



# Chemin vers votre fichier texte
chemin_fichier = "data.txt"

# Lire le fichier texte et créer un DataFrame pandas
df = pd.read_csv(chemin_fichier, delimiter=' ', header=None)
df.columns = ['nom','type','gant','force','cond','bloc1', 'bloc2','bloc3', 'bloc4','bloc5', 'bloc6','bloc7', 'bloc8','bloc9', 'bloc10','bloc11', 'bloc12','bloc13', 'bloc14','bloc15', 'bloc16','bloc17', 'bloc18','bloc19', 'bloc20']

# Afficher le DataFrame
print(df)

nom = ['Victor', 'Lise', 'Sophie', 'Hugo']


#comparer le up et down de la reproduction lf
moyenne_question1 = []
m_down_rep_a =[]
m_up_rep_a =[]
m_down_rep_s =[]
m_up_rep_s =[]
for n in nom :
    moyenne = []
    fig = plt.figure(figsize = [15,9])
    
    down_rep_a = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='a')].iloc[:, 5:17:2] #down avec gants
    down_rep_a = down_rep_a.dropna()
    list_down_rep_a = down_rep_a.values.flatten()
    moyenne_down_rep_a = np.mean(list_down_rep_a)
    moyenne.append(moyenne_down_rep_a)
    m_down_rep_a.append(moyenne_down_rep_a)

    up_rep_a = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='a')].iloc[:, 6:17:2] #up avec gants
    up_rep_a = up_rep_a.dropna()
    list_up_rep_a = up_rep_a.values.flatten()
    moyenne_up_rep_a = np.mean(list_up_rep_a)
    moyenne.append(moyenne_up_rep_a)
    m_up_rep_a.append(moyenne_up_rep_a)

    down_rep_s = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='s')].iloc[:, 5:17:2] #down sans gants
    down_rep_s = down_rep_a.dropna()
    list_down_rep_s = down_rep_s.values.flatten()
    moyenne_down_rep_s = np.mean(list_down_rep_s)
    moyenne.append(moyenne_down_rep_s)
    m_down_rep_s.append(moyenne_down_rep_s)

    up_rep_s = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='s')].iloc[:, 6:17:2] #up sans gants
    up_rep_s = up_rep_s.dropna()
    list_up_rep_s = up_rep_s.values.flatten()
    moyenne_up_rep_s = np.mean(list_up_rep_s)
    moyenne.append(moyenne_up_rep_s)
    m_up_rep_s.append(moyenne_up_rep_s)

    moyenne_question1.append(moyenne)


    xs_down_rep_a = []
    for i in range(len(list_down_rep_a)):
        xs_down_rep_a.append(np.random.normal((i+1)*10e-20,0.02))

    xs_up_rep_a = []
    for i in range(len(list_up_rep_a)):
        xs_up_rep_a.append(np.random.normal((i+1)*10e-20 +2,0.02))

    xs_down_rep_s = []
    for i in range(len(list_down_rep_s)):
        xs_down_rep_s.append(np.random.normal((i+1)*10e-20+1,0.02))

    xs_up_rep_s = []
    for i in range(len(list_up_rep_s)):
        xs_up_rep_s.append(np.random.normal((i+1)*10e-20 +3,0.02))

    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_a, positions=[0],labels=['bas avec gants'])
    for xs,val in zip(xs_down_rep_a, list_down_rep_a):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_a, positions=[2],labels=['haut avec gants'])
    for xs,val in zip(xs_up_rep_a,list_up_rep_a):
        plt.scatter(xs, val, color='green')
    
    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_s, positions=[1],labels=['bas sans gants'])
    for xs,val in zip(xs_down_rep_s, list_down_rep_s):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_s, positions=[3],labels=['haut sans gants'])
    for xs,val in zip(xs_up_rep_s,list_up_rep_s):
        plt.scatter(xs, val, color='green')

    plt.grid()
    plt.title(n+" load force : reproduction")
    plt.show()

##global##
fig = plt.figure(figsize = [15,9])
    
down_rep_a = df[( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='a')].iloc[:, 5:17:2] #down avec gants
down_rep_a = down_rep_a.dropna()
list_down_rep_a = down_rep_a.values.flatten()

up_rep_a = df[(df['type']=='rep')&(df['force']=='lf')&(df['gant']=='a')].iloc[:, 6:17:2] #up avec gants
up_rep_a = up_rep_a.dropna()
list_up_rep_a = up_rep_a.values.flatten()

down_rep_s = df[( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='s')].iloc[:, 5:17:2] #down sans gants
down_rep_s = down_rep_a.dropna()
list_down_rep_s = down_rep_s.values.flatten()

up_rep_s = df[( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='s')].iloc[:, 6:17:2] #up sans gants
up_rep_s = up_rep_s.dropna()
list_up_rep_s = up_rep_s.values.flatten()



xs_down_rep_a = []
for i in range(len(list_down_rep_a)):
    xs_down_rep_a.append(np.random.normal((i+1)*10e-20,0.02))

xs_up_rep_a = []
for i in range(len(list_up_rep_a)):
    xs_up_rep_a.append(np.random.normal((i+1)*10e-20 +2,0.02))

xs_down_rep_s = []
for i in range(len(list_down_rep_s)):
    xs_down_rep_s.append(np.random.normal((i+1)*10e-20+1,0.02))

xs_up_rep_s = []
for i in range(len(list_up_rep_s)):
    xs_up_rep_s.append(np.random.normal((i+1)*10e-20 +3,0.02))

#create box plots for the even indexes of the line
plt.boxplot(list_down_rep_a, positions=[0],labels=['bas avec gants'])
for xs,val in zip(xs_down_rep_a, list_down_rep_a):
    plt.scatter(xs, val, color='red')

#create box plots for the odd indexes of the line
plt.boxplot(list_up_rep_a, positions=[2],labels=['haut avec gants'])
for xs,val in zip(xs_up_rep_a,list_up_rep_a):
    plt.scatter(xs, val, color='green')

#create box plots for the even indexes of the line
plt.boxplot(list_down_rep_s, positions=[1],labels=['bas sans gants'])
for xs,val in zip(xs_down_rep_s, list_down_rep_s):
    plt.scatter(xs, val, color='red')

#create box plots for the odd indexes of the line
plt.boxplot(list_up_rep_s, positions=[3],labels=['haut sans gants'])
for xs,val in zip(xs_up_rep_s,list_up_rep_s):
    plt.scatter(xs, val, color='green')

plt.grid()
plt.title("Load force reproduction globale")
plt.show()

x = [1,2,3,4]
m =[]
for i in range (4):
    moyenne = (moyenne_question1[0][i] + moyenne_question1[1][i] + moyenne_question1[2][i] + moyenne_question1[3][i])/4
    m.append(moyenne)
error_down_rep_a = np.std(m_down_rep_a, ddof=1) / np.sqrt(len(m_down_rep_a))
error_up_rep_a = np.std(m_up_rep_a, ddof=1) / np.sqrt(len(m_up_rep_a))
error_down_rep_s = np.std(m_down_rep_s, ddof=1) / np.sqrt(len(m_down_rep_s))
error_up_rep_s = np.std(m_up_rep_s, ddof=1) / np.sqrt(len(m_up_rep_s))

plt.scatter(x, moyenne_question1[0], color='red', label='Victor')
plt.scatter(x, moyenne_question1[1], color='blue', label='Lise')
plt.scatter(x, moyenne_question1[2], color='green', label='Sophie')
plt.scatter(x, moyenne_question1[3], color='orange', label='Hugo')
plt.scatter(x, m, color='black', label='Moyenne totale')
plt.errorbar(1, m[0], yerr=error_down_rep_a, fmt='o', capsize=5, color='black', label='Erreur standard')
plt.errorbar(2, m[1], yerr=error_up_rep_a, fmt='o', capsize=5, color='black')
plt.errorbar(3, m[2], yerr=error_down_rep_s, fmt='o', capsize=5, color='black')
plt.errorbar(4, m[3], yerr=error_up_rep_s, fmt='o', capsize=5, color='black')
plt.legend()
plt.ylim(3, 9)
plt.ylabel('LF [N]')
plt.xticks([1, 2, 3, 4], ['bas avec gants', 'haut avec gants', 'bas sans gants', 'haut avec gants'])
plt.title("Load force reproduction : moyenne")
plt.show()


# Créer un DataFrame avec les données
datatest = pd.DataFrame({
    'Force': m_down_rep_a + m_up_rep_a + m_down_rep_s + m_up_rep_s,  
    'Participant': nom * 4,
    'Direction': ['down'] * 4 + ['up'] * 4 + ['down'] * 4 + ['up'] * 4,
    'Gants': ['Oui'] * 8  + ['Non'] * 8
})

# Convertir les variables 'Direction', 'Gants' et 'Participant' en variables catégoriques
datatest['Direction'] = pd.Categorical(datatest['Direction'])
datatest['Gants'] = pd.Categorical(datatest['Gants'])
datatest['Participant'] = pd.Categorical(datatest['Participant'])

# Effectuer l'ANOVA à mesures répétées
anova = AnovaRM(datatest, 'Force', 'Participant', within=['Direction', 'Gants'], aggregate_func='mean').fit()

# Afficher les résultats de l'ANOVA à mesures répétées
print(anova.summary())
#Si la valeur p associée à la statistique de test F est inférieure à un seuil spécifié (généralement 0,05),
#on rejette l'hypothèse nulle et on conclut qu'il existe une différence statistiquement significative entre les moyennes des groupes.
#direction : gants = interaction entre les deux variables
# p-valeur = valeur de la dernière colonne 


#comparer si reproduction et entrainement identique quand condition gant identique ou pas (et différencier up et down) lf

#conditions identiques
"""
for n in nom :
    fig = plt.figure(figsize = [15,9])

    down_entr_id = df[(df['nom'] == n )&( df['type']=='entr')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 5:17:2] #down conditions identiques entrainement
    down_entr_id = down_entr_id.dropna()
    list_down_entr_id = down_entr_id.values.flatten()

    up_entr_id = df[(df['nom'] == n)&( df['type']=='entr')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 6:17:2] #up conditions identiques entrainement
    up_entr_id = up_entr_id.dropna()
    list_up_entr_id = up_entr_id.values.flatten()

    down_rep_id = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 5:17:2] #down conditions identiques reproduction
    down_rep_id = down_rep_id.dropna()
    list_down_rep_id = down_rep_id.values.flatten()

    up_rep_id = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 6:17:2] #up conditions identiques reproduction
    up_rep_id = up_rep_id.dropna()
    list_up_rep_id = up_rep_id.values.flatten()

    xs_down_entr_id = []
    for i in range(len(list_down_entr_id)):
        xs_down_entr_id.append(np.random.normal((i+1)*10e-20,0.02))

    xs_up_entr_id = []
    for i in range(len(list_up_entr_id)):
        xs_up_entr_id.append(np.random.normal((i+1)*10e-20+2,0.02))

    xs_down_rep_id = []
    for i in range(len(list_down_rep_id)):
        xs_down_rep_id.append(np.random.normal((i+1)*10e-20+1,0.02))

    xs_up_rep_id = []
    for i in range(len(list_up_rep_id)):
        xs_up_rep_id.append(np.random.normal((i+1)*10e-20+3,0.02))


    #create box plots for the even indexes of the line
    plt.boxplot(list_down_entr_id, positions=[0],labels=['down entr id'])
    for xs,val in zip(xs_down_entr_id, list_down_entr_id):
        plt.scatter(xs, val, color='red')

    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_id, positions=[1],labels=['down rep id'])
    for xs,val in zip(xs_down_rep_id, list_down_rep_id):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_entr_id, positions=[2],labels=['up entr id'])
    for xs,val in zip(xs_up_entr_id, list_up_entr_id):
        plt.scatter(xs, val, color='green')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_id, positions=[3],labels=['up rep id'])
    for xs,val in zip(xs_up_rep_id, list_up_rep_id):
        plt.scatter(xs, val, color='green')

    plt.grid()
    plt.title(n +" LF conditions identiques entr et rep")
    plt.show()

##global##

fig = plt.figure(figsize = [15,9])

down_entr_id = df[( df['type']=='entr')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 5:17:2] #down conditions identiques entrainement
down_entr_id = down_entr_id.dropna()
list_down_entr_id = down_entr_id.values.flatten()

up_entr_id = df[( df['type']=='entr')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 6:17:2] #up conditions identiques entrainement
up_entr_id = up_entr_id.dropna()
list_up_entr_id = up_entr_id.values.flatten()

down_rep_id = df[( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 5:17:2] #down conditions identiques reproduction
down_rep_id = down_rep_id.dropna()
list_down_rep_id = down_rep_id.values.flatten()

up_rep_id = df[( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 6:17:2] #up conditions identiques reproduction
up_rep_id = up_rep_id.dropna()
list_up_rep_id = up_rep_id.values.flatten()


xs_down_entr_id = []
for i in range(len(list_down_entr_id)):
    xs_down_entr_id.append(np.random.normal((i+1)*10e-20,0.02))

xs_up_entr_id = []
for i in range(len(list_up_entr_id)):
    xs_up_entr_id.append(np.random.normal((i+1)*10e-20+2,0.02))

xs_down_rep_id = []
for i in range(len(list_down_rep_id)):
    xs_down_rep_id.append(np.random.normal((i+1)*10e-20+1,0.02))

xs_up_rep_id = []
for i in range(len(list_up_rep_id)):
    xs_up_rep_id.append(np.random.normal((i+1)*10e-20+3,0.02))


#create box plots for the even indexes of the line
plt.boxplot(list_down_entr_id, positions=[0],labels=['down entr id'])
for xs,val in zip(xs_down_entr_id, list_down_entr_id): 
    plt.scatter(xs, val, color='red')

#create box plots for the even indexes of the line
plt.boxplot(list_down_rep_id, positions=[1],labels=['down rep id'])
for xs,val in zip(xs_down_rep_id, list_down_rep_id):
    plt.scatter(xs, val, color='red')

#create box plots for the odd indexes of the line
plt.boxplot(list_up_entr_id, positions=[2],labels=['up entr id'])
for xs,val in zip(xs_up_entr_id, list_up_entr_id):
    plt.scatter(xs, val, color='green')

#create box plots for the odd indexes of the line
plt.boxplot(list_up_rep_id, positions=[3],labels=['up rep id'])
for xs,val in zip(xs_up_rep_id, list_up_rep_id):
    plt.scatter(xs, val, color='green')

plt.grid()
plt.title("global LF conditions identiques entr et rep")
plt.show()
"""
"""
#conditions différentes
for n in nom :
    fig = plt.figure(figsize = [15,9])

    down_entr_dif = df[(df['nom'] == n )&( df['type']=='entr')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 5:17:2] #down conditions identiques entrainement
    down_entr_dif = down_entr_dif.dropna()
    list_down_entr_dif = down_entr_dif.values.flatten()

    up_entr_dif = df[(df['nom'] == n)&( df['type']=='entr')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 6:17:2] #up conditions identiques entrainement
    up_entr_dif = up_entr_dif.dropna()
    list_up_entr_dif = up_entr_dif.values.flatten()

    down_rep_dif = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 5:17:2] #down conditions identiques reproduction
    down_rep_dif = down_rep_dif.dropna()
    list_down_rep_dif = down_rep_dif.values.flatten()

    up_rep_dif = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 6:17:2] #up conditions identiques reproduction
    up_rep_dif = up_rep_dif.dropna()
    list_up_rep_dif = up_rep_dif.values.flatten()

    xs_down_entr_dif = []
    for i in range(len(list_down_entr_dif)):
        xs_down_entr_dif.append(np.random.normal((i+1)*10e-20,0.02))

    xs_up_entr_dif = []
    for i in range(len(list_up_entr_dif)):
        xs_up_entr_dif.append(np.random.normal((i+1)*10e-20+2,0.02))

    xs_down_rep_dif = []
    for i in range(len(list_down_rep_dif)):
        xs_down_rep_dif.append(np.random.normal((i+1)*10e-20+1,0.02))

    xs_up_rep_dif= []
    for i in range(len(list_up_rep_dif)):
        xs_up_rep_dif.append(np.random.normal((i+1)*10e-20+3,0.02))


    #create box plots for the even indexes of the line
    plt.boxplot(list_down_entr_dif, positions=[0],labels=['down entr dif'])
    for xs,val in zip(xs_down_entr_dif, list_down_entr_dif):
        plt.scatter(xs, val, color='red')

    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_dif, positions=[1],labels=['down rep dif'])
    for xs,val in zip(xs_down_rep_dif, list_down_rep_dif):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_entr_dif, positions=[2],labels=['up entr dif'])
    for xs,val in zip(xs_up_entr_dif, list_up_entr_dif):
        plt.scatter(xs, val, color='green')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_dif, positions=[3],labels=['up rep dif'])
    for xs,val in zip(xs_up_rep_dif, list_up_rep_dif):
        plt.scatter(xs, val, color='green')

    plt.grid()
    plt.title(n +" LF conditions différentes entr et rep")
    plt.show()

##global##
fig = plt.figure(figsize = [15,9])

down_entr_dif = df[( df['type']=='entr')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 5:17:2] #down conditions différentes entrainement
down_entr_dif = down_entr_dif.dropna()
list_down_entr_dif = down_entr_dif.values.flatten()

up_entr_dif = df[( df['type']=='entr')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 6:17:2] #up conditions différentes entrainement
up_entr_dif = up_entr_dif.dropna()
list_up_entr_dif = up_entr_dif.values.flatten()

down_rep_dif = df[( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 5:17:2] #down conditions différentes reproduction
down_rep_dif = down_rep_dif.dropna()
list_down_rep_dif = down_rep_dif.values.flatten()

up_rep_dif = df[( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 6:17:2] #up conditions différentes reproduction
up_rep_dif = up_rep_dif.dropna()
list_up_rep_dif = up_rep_dif.values.flatten()

xs_down_entr_dif = []
for i in range(len(list_down_entr_dif)):
    xs_down_entr_dif.append(np.random.normal((i+1)*10e-20,0.02))

xs_up_entr_dif = []
for i in range(len(list_up_entr_dif)):
    xs_up_entr_dif.append(np.random.normal((i+1)*10e-20+2,0.02))

xs_down_rep_dif = []
for i in range(len(list_down_rep_dif)):
    xs_down_rep_dif.append(np.random.normal((i+1)*10e-20+1,0.02))

xs_up_rep_dif= []
for i in range(len(list_up_rep_dif)):
    xs_up_rep_dif.append(np.random.normal((i+1)*10e-20+3,0.02))


#create box plots for the even indexes of the line
plt.boxplot(list_down_entr_dif, positions=[0],labels=['down entr dif'])
for xs,val in zip(xs_down_entr_dif, list_down_entr_dif):
    plt.scatter(xs, val, color='red')

#create box plots for the even indexes of the line
plt.boxplot(list_down_rep_dif, positions=[1],labels=['down rep dif'])
for xs,val in zip(xs_down_rep_dif, list_down_rep_dif):
    plt.scatter(xs, val, color='red')

#create box plots for the odd indexes of the line
plt.boxplot(list_up_entr_dif, positions=[2],labels=['up entr dif'])
for xs,val in zip(xs_up_entr_dif, list_up_entr_dif):
    plt.scatter(xs, val, color='green')

#create box plots for the odd indexes of the line
plt.boxplot(list_up_rep_dif, positions=[3],labels=['up rep dif'])
for xs,val in zip(xs_up_rep_dif, list_up_rep_dif):
    plt.scatter(xs, val, color='green')

plt.grid()
plt.title("global LF conditions différentes entr et rep")
plt.show()
"""


#comparer si reproduction lf egale quand conditions ident et diff
"""
for n in nom :
    fig = plt.figure(figsize = [15,9])

    down_rep_id = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 5:17:2] #down conditions identiques reproduction
    down_rep_id = down_rep_id.dropna()
    list_down_rep_id = down_rep_id.values.flatten()

    up_rep_id = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 6:17:2] #up conditions identiques reproduction
    up_rep_id = up_rep_id.dropna()
    list_up_rep_id = up_rep_id.values.flatten()

    down_rep_dif = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 5:17:2] #down conditions différentes reproduction
    down_rep_dif = down_rep_dif.dropna()
    list_down_rep_dif = down_rep_dif.values.flatten()

    up_rep_dif = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 6:17:2] #up conditions différentes reproduction
    up_rep_dif = up_rep_dif.dropna()
    list_up_rep_dif = up_rep_dif.values.flatten()

    xs_down_rep_id = []
    for i in range(len(list_down_rep_id)):
        xs_down_rep_id.append(np.random.normal((i+1)*10e-20,0.02))

    xs_up_rep_id = []
    for i in range(len(list_up_rep_id)):
        xs_up_rep_id.append(np.random.normal((i+1)*10e-20+2,0.02))

    xs_down_rep_dif = []
    for i in range(len(list_down_rep_dif)):
        xs_down_rep_dif.append(np.random.normal((i+1)*10e-20+1,0.02))

    xs_up_rep_dif= []
    for i in range(len(list_up_rep_dif)):
        xs_up_rep_dif.append(np.random.normal((i+1)*10e-20+3,0.02))


    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_id, positions=[0],labels=['down rep id'])
    for xs,val in zip(xs_down_rep_id, list_down_rep_id):
        plt.scatter(xs, val, color='red')

    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_dif, positions=[1],labels=['down rep dif'])
    for xs,val in zip(xs_down_rep_dif, list_down_rep_dif):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_id, positions=[2],labels=['up rep id'])
    for xs,val in zip(xs_up_rep_id, list_up_rep_id):
        plt.scatter(xs, val, color='green')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_dif, positions=[3],labels=['up rep dif'])
    for xs,val in zip(xs_up_rep_dif, list_up_rep_dif):
        plt.scatter(xs, val, color='green')

    plt.grid()
    plt.title(n +" LF rep conditions ident et diff")
    plt.show()

##global##

fig = plt.figure(figsize = [15,9])

down_rep_id = df[( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 5:17:2] #down conditions identiques reproduction
down_rep_id = down_rep_id.dropna()
list_down_rep_id = down_rep_id.values.flatten()

up_rep_id = df[(df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 6:17:2] #up conditions identiques reproduction
up_rep_id = up_rep_id.dropna()
list_up_rep_id = up_rep_id.values.flatten()

down_rep_dif = df[(df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 5:17:2] #down conditions différentes reproduction
down_rep_dif = down_rep_dif.dropna()
list_down_rep_dif = down_rep_dif.values.flatten()

up_rep_dif = df[(df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 6:17:2] #up conditions différentes reproduction
up_rep_dif = up_rep_dif.dropna()
list_up_rep_dif = up_rep_dif.values.flatten()

xs_down_rep_id = []
for i in range(len(list_down_rep_id)):
    xs_down_rep_id.append(np.random.normal((i+1)*10e-20,0.02))

xs_up_rep_id = []
for i in range(len(list_up_rep_id)):
    xs_up_rep_id.append(np.random.normal((i+1)*10e-20+2,0.02))

xs_down_rep_dif = []
for i in range(len(list_down_rep_dif)):
    xs_down_rep_dif.append(np.random.normal((i+1)*10e-20+1,0.02))

xs_up_rep_dif= []
for i in range(len(list_up_rep_dif)):
    xs_up_rep_dif.append(np.random.normal((i+1)*10e-20+3,0.02))


#create box plots for the even indexes of the line
plt.boxplot(list_down_rep_id, positions=[0],labels=['down rep id'])
for xs,val in zip(xs_down_rep_id, list_down_rep_id):
    plt.scatter(xs, val, color='red')

#create box plots for the even indexes of the line
plt.boxplot(list_down_rep_dif, positions=[1],labels=['down rep dif'])
for xs,val in zip(xs_down_rep_dif, list_down_rep_dif):
    plt.scatter(xs, val, color='red')

#create box plots for the odd indexes of the line
plt.boxplot(list_up_rep_id, positions=[2],labels=['up rep id'])
for xs,val in zip(xs_up_rep_id, list_up_rep_id):
    plt.scatter(xs, val, color='green')

#create box plots for the odd indexes of the line
plt.boxplot(list_up_rep_dif, positions=[3],labels=['up rep dif'])
for xs,val in zip(xs_up_rep_dif, list_up_rep_dif):
    plt.scatter(xs, val, color='green')

plt.grid()
plt.title(n +" LF rep conditions ident et diff")
plt.show()
"""

"""
#comparer les rapports grip force et load force avec et sans gants (et différencier up et down)
#LF<=GF*µ -> on va regarder µ*GF/LF

for n in nom :
    fig = plt.figure(figsize = [15,9])
    mu_sans = 0.8
    mu_avec = 0.6

    #avec gants
    down_rep_a_lf = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='a')].iloc[:, 5:17:2] #down avec gant reproduction lf
    down_rep_a_lf  = down_rep_a_lf .dropna()
    list_down_rep_a_lf = down_rep_a_lf .values.flatten()

    up_rep_a_lf= df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='a')].iloc[:, 6:17:2] #up avec gant reproduction lf
    up_rep_a_lf = up_rep_a_lf.dropna()
    list_up_rep_a_lf= up_rep_a_lf.values.flatten()

    down_rep_a_gf = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='gf')&(df['gant']=='a')].iloc[:, 5:17:2] #down avec gant reproduction gf
    down_rep_a_gf = down_rep_a_gf.dropna()
    list_down_rep_a_gf = down_rep_a_gf.values.flatten()

    up_rep_a_gf = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='gf')&(df['gant']=='a')].iloc[:, 6:17:2] #up avec gant reproduction gf
    up_rep_a_gf = up_rep_a_gf.dropna()
    list_up_rep_a_gf = up_rep_a_gf.values.flatten()

    #calcul des rapports
    rapport_down_a = []
    for i in range(len(list_down_rep_a_lf)):
        rapport_down_a.append(list_down_rep_a_gf[i]*mu_avec/list_down_rep_a_lf[i])
    rapport_up_a = []
    for i in range(len(list_up_rep_a_lf)):
        rapport_up_a.append(list_up_rep_a_gf[i]*mu_avec/list_up_rep_a_lf[i])

    #sans gants
    down_rep_s_lf = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='s')].iloc[:, 5:17:2] #down sans gant reproduction lf
    down_rep_s_lf  = down_rep_s_lf .dropna()
    list_down_rep_s_lf = down_rep_s_lf .values.flatten()

    up_rep_s_lf= df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['gant']=='s')].iloc[:, 6:17:2] #up avec sans reproduction lf
    up_rep_s_lf = up_rep_s_lf.dropna()
    list_up_rep_s_lf= up_rep_s_lf.values.flatten()

    down_rep_s_gf = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='gf')&(df['gant']=='s')].iloc[:, 5:17:2] #down sans gant reproduction gf
    down_rep_s_gf = down_rep_a_gf.dropna()
    list_down_rep_s_gf = down_rep_s_gf.values.flatten()

    up_rep_s_gf = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='gf')&(df['gant']=='s')].iloc[:, 6:17:2] #up avec sans reproduction gf
    up_rep_s_gf = up_rep_s_gf.dropna()
    list_up_rep_s_gf = up_rep_s_gf.values.flatten()

    #calcul des rapports
    rapport_down_s = []
    for i in range(len(list_down_rep_s_lf)):
        rapport_down_s.append(list_down_rep_s_gf[i]*mu_sans/list_down_rep_s_lf[i])
    rapport_up_s = []
    for i in range(len(list_up_rep_s_lf)):
        rapport_up_s.append(list_up_rep_s_gf[i]*mu_sans/list_up_rep_s_lf[i])


    xs_down_a = []
    for i in range(len(rapport_down_a)):
        xs_down_a.append(np.random.normal((i+1)*10e-20,0.02))

    xs_up_a = []
    for i in range(len(rapport_up_a)):
        xs_up_a.append(np.random.normal((i+1)*10e-20+2,0.02))

    xs_down_s = []
    for i in range(len(rapport_down_s)):
        xs_down_s.append(np.random.normal((i+1)*10e-20+1,0.02))

    xs_up_s= []
    for i in range(len(rapport_up_s)):
        xs_up_s.append(np.random.normal((i+1)*10e-20+3,0.02))


    #create box plots for the even indexes of the line
    plt.boxplot(rapport_down_a, positions=[0],labels=['down avec gant'])
    for xs,val in zip(xs_down_a, rapport_down_a):
        plt.scatter(xs, val, color='red')

    #create box plots for the even indexes of the line
    plt.boxplot(rapport_down_s, positions=[1],labels=['down sans gant'])
    for xs,val in zip(xs_down_s, rapport_down_s):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(rapport_up_a, positions=[2],labels=['up avec gant'])
    for xs,val in zip(xs_up_a, rapport_up_a):
        plt.scatter(xs, val, color='green')

    #create box plots for the odd indexes of the line
    plt.boxplot(rapport_up_s, positions=[3],labels=['up sans gant'])
    for xs,val in zip(xs_up_s, rapport_up_s):
        plt.scatter(xs, val, color='green')

    plt.grid()
    plt.title(n +" rapport LF et GF avec et sans gants")
    plt.show()

"""