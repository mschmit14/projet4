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
<<<<<<< HEAD
partie = 1
=======
partie = 'partie5'
>>>>>>> c49e1426ebd7a52112008d9d94d526562700856b


if partie == 'partie0':
    #grip force de reproduction avec gant (pour voir le minimum pour calculer le coeffcient de frottement avec gant)
    moyenne_question1 = []
    m_down_rep_a =[]
    m_up_rep_a =[]
    m_down_rep_s =[]
    m_up_rep_s =[]
    for n in nom :
        moyenne = []
        fig = plt.figure(figsize = [15,9])
        
        down_rep_a = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='gf')&(df['gant']=='a')].iloc[:, 5:17:2] #down avec gants
        down_rep_a = down_rep_a.dropna()
        list_down_rep_a = down_rep_a.values.flatten()
        moyenne_down_rep_a = np.mean(list_down_rep_a)
        moyenne.append(moyenne_down_rep_a)
        m_down_rep_a.append(moyenne_down_rep_a)

        up_rep_a = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='gf')&(df['gant']=='a')].iloc[:, 6:17:2] #up avec gants
        up_rep_a = up_rep_a.dropna()
        list_up_rep_a = up_rep_a.values.flatten()
        moyenne_up_rep_a = np.mean(list_up_rep_a)
        moyenne.append(moyenne_up_rep_a)
        m_up_rep_a.append(moyenne_up_rep_a)

        moyenne_question1.append(moyenne)


        xs_down_rep_a = []
        for i in range(len(list_down_rep_a)):
            xs_down_rep_a.append(np.random.normal((i+1)*10e-20,0.02))

        xs_up_rep_a = []
        for i in range(len(list_up_rep_a)):
            xs_up_rep_a.append(np.random.normal((i+1)*10e-20 +1,0.02))

        #create box plots for the even indexes of the line
        plt.boxplot(list_down_rep_a, positions=[0],labels=['bas avec gant'])
        for xs,val in zip(xs_down_rep_a, list_down_rep_a):
            plt.scatter(xs, val, color='red')

        #create box plots for the odd indexes of the line
        plt.boxplot(list_up_rep_a, positions=[1],labels=['haut avec gant'])
        for xs,val in zip(xs_up_rep_a,list_up_rep_a):
            plt.scatter(xs, val, color='green')
        plt.grid()
        plt.title(n+" GF reproduction avec gant")
        plt.show()

    ##global##
    fig = plt.figure(figsize = [15,9])
        
    down_rep_a = df[( df['type']=='rep')&(df['force']=='gf')&(df['gant']=='a')].iloc[:, 5:17:2] #down avec gants
    down_rep_a = down_rep_a.dropna()
    list_down_rep_a = down_rep_a.values.flatten()

    up_rep_a = df[(df['type']=='rep')&(df['force']=='gf')&(df['gant']=='a')].iloc[:, 6:17:2] #up avec gants
    up_rep_a = up_rep_a.dropna()
    list_up_rep_a = up_rep_a.values.flatten()

    xs_down_rep_a = []
    for i in range(len(list_down_rep_a)):
        xs_down_rep_a.append(np.random.normal((i+1)*10e-20,0.02))

    xs_up_rep_a = []
    for i in range(len(list_up_rep_a)):
        xs_up_rep_a.append(np.random.normal((i+1)*10e-20 +1,0.02))

   

    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_a, positions=[0],labels=['bas avec gant'])
    for xs,val in zip(xs_down_rep_a, list_down_rep_a):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_a, positions=[1],labels=['haut avec gant'])
    for xs,val in zip(xs_up_rep_a,list_up_rep_a):
        plt.scatter(xs, val, color='green')

    plt.grid()
    plt.title("global GF reproduction avec gant")
    plt.show()

    x = [1,2]
    m =[]
    for i in range (2):
        moyenne = (moyenne_question1[0][i] + moyenne_question1[1][i] )/2
        m.append(moyenne)
    error_down_rep_a = np.std(m_down_rep_a, ddof=1) / np.sqrt(len(m_down_rep_a))
    error_up_rep_a = np.std(m_up_rep_a, ddof=1) / np.sqrt(len(m_up_rep_a))
   

    plt.scatter(x, moyenne_question1[0], color='red', label='Victor')
    plt.scatter(x, moyenne_question1[1], color='blue', label='Lise')
    plt.scatter(x, moyenne_question1[2], color='green', label='Sophie')
    plt.scatter(x, moyenne_question1[3], color='orange', label='Hugo')
    plt.scatter(x, m, color='black', label='Moyenne totale')
    plt.errorbar(1, m[0], yerr=error_down_rep_a, fmt='o', capsize=5, color='black', label='Erreur standard')
    plt.errorbar(2, m[1], yerr=error_up_rep_a, fmt='o', capsize=5, color='black')
   
    plt.legend()
    plt.ylim(3, 9)
    plt.ylabel('LF [N]')
    plt.xticks([1, 2], ['bas avec gant', 'haut avec gant'])
    plt.title("Moyenne LF reproduction avec et sans gant")
    plt.show()


if partie == 'partie1':
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
            xs_up_rep_a.append(np.random.normal((i+1)*10e-20 +1,0.02))

        xs_down_rep_s = []
        for i in range(len(list_down_rep_s)):
            xs_down_rep_s.append(np.random.normal((i+1)*10e-20+2,0.02))

        xs_up_rep_s = []
        for i in range(len(list_up_rep_s)):
            xs_up_rep_s.append(np.random.normal((i+1)*10e-20 +3,0.02))

        #create box plots for the even indexes of the line
        plt.boxplot(list_down_rep_a, positions=[0],labels=['bas avec gant'])
        for xs,val in zip(xs_down_rep_a, list_down_rep_a):
            plt.scatter(xs, val, color='red')

        #create box plots for the odd indexes of the line
        plt.boxplot(list_up_rep_a, positions=[1],labels=['haut avec gant'])
        for xs,val in zip(xs_up_rep_a,list_up_rep_a):
            plt.scatter(xs, val, color='green')
        
        #create box plots for the even indexes of the line
        plt.boxplot(list_down_rep_s, positions=[2],labels=['bas sans gant'])
        for xs,val in zip(xs_down_rep_s, list_down_rep_s):
            plt.scatter(xs, val, color='red')

        #create box plots for the odd indexes of the line
        plt.boxplot(list_up_rep_s, positions=[3],labels=['haut sans gant'])
        for xs,val in zip(xs_up_rep_s,list_up_rep_s):
            plt.scatter(xs, val, color='green')

        plt.grid()
        plt.title(n+" LF reproduction")
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
        xs_up_rep_a.append(np.random.normal((i+1)*10e-20 +1,0.02))

    xs_down_rep_s = []
    for i in range(len(list_down_rep_s)):
        xs_down_rep_s.append(np.random.normal((i+1)*10e-20+2,0.02))

    xs_up_rep_s = []
    for i in range(len(list_up_rep_s)):
        xs_up_rep_s.append(np.random.normal((i+1)*10e-20 +3,0.02))

    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_a, positions=[0],labels=['bas avec gant'])
    for xs,val in zip(xs_down_rep_a, list_down_rep_a):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_a, positions=[1],labels=['haut avec gant'])
    for xs,val in zip(xs_up_rep_a,list_up_rep_a):
        plt.scatter(xs, val, color='green')

    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_s, positions=[2],labels=['bas sans gant'])
    for xs,val in zip(xs_down_rep_s, list_down_rep_s):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_s, positions=[3],labels=['haut sans gant'])
    for xs,val in zip(xs_up_rep_s,list_up_rep_s):
        plt.scatter(xs, val, color='green')

    plt.grid()
    plt.title("global LF reproduction")
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
    plt.xticks([1, 2, 3, 4], ['bas avec gant', 'haut avec gant', 'bas sans gant', 'haut sans gant'])
    plt.title("Moyenne LF reproduction avec et sans gant")
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

if partie == 'partie2':
    #comparer si reproduction lf egale quand conditions ident et diff
    moyenne_question2 = []
    m_down_rep_id =[]
    m_up_rep_id =[]
    m_down_rep_dif =[]
    m_up_rep_dif =[]

    for n in nom :
        fig = plt.figure(figsize = [15,9])
        moyenne = []

        down_rep_id = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 5:17:2] #down conditions identiques reproduction
        down_rep_id = down_rep_id.dropna()
        list_down_rep_id = down_rep_id.values.flatten()
        moyenne_down_rep_id = np.mean(list_down_rep_id)
        moyenne.append(moyenne_down_rep_id)
        m_down_rep_id.append(moyenne_down_rep_id)

        up_rep_id = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')].iloc[:, 6:17:2] #up conditions identiques reproduction
        up_rep_id = up_rep_id.dropna()
        list_up_rep_id = up_rep_id.values.flatten()
        moyenne_up_rep_id = np.mean(list_up_rep_id)
        moyenne.append(moyenne_up_rep_id)
        m_up_rep_id.append(moyenne_up_rep_id)

        down_rep_dif = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 5:17:2] #down conditions différentes reproduction
        down_rep_dif = down_rep_dif.dropna()
        list_down_rep_dif = down_rep_dif.values.flatten()
        moyenne_down_rep_dif = np.mean(list_down_rep_dif)
        moyenne.append(moyenne_down_rep_dif)
        m_down_rep_dif.append(moyenne_down_rep_dif)

        up_rep_dif = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')].iloc[:, 6:17:2] #up conditions différentes reproduction
        up_rep_dif = up_rep_dif.dropna()
        list_up_rep_dif = up_rep_dif.values.flatten()
        moyenne_up_rep_dif = np.mean(list_up_rep_dif)
        moyenne.append(moyenne_up_rep_dif)
        m_up_rep_dif.append(moyenne_up_rep_dif)

        moyenne_question2.append(moyenne)

        xs_down_rep_id = []
        for i in range(len(list_down_rep_id)):
            xs_down_rep_id.append(np.random.normal((i+1)*10e-20,0.02))

        xs_up_rep_id = []
        for i in range(len(list_up_rep_id)):
            xs_up_rep_id.append(np.random.normal((i+1)*10e-20+1,0.02))

        xs_down_rep_dif = []
        for i in range(len(list_down_rep_dif)):
            xs_down_rep_dif.append(np.random.normal((i+1)*10e-20+2,0.02))

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
        plt.title(n +" LF rep conditions identiques et différentes")
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
        xs_up_rep_id.append(np.random.normal((i+1)*10e-20+1,0.02))

    xs_down_rep_dif = []
    for i in range(len(list_down_rep_dif)):
        xs_down_rep_dif.append(np.random.normal((i+1)*10e-20+2,0.02))

    xs_up_rep_dif= []
    for i in range(len(list_up_rep_dif)):
        xs_up_rep_dif.append(np.random.normal((i+1)*10e-20+3,0.02))


    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_id, positions=[0],labels=['bas rep id'])
    for xs,val in zip(xs_down_rep_id, list_down_rep_id):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_id, positions=[1],labels=['haut rep id'])
    for xs,val in zip(xs_up_rep_id, list_up_rep_id):
        plt.scatter(xs, val, color='green')

    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_dif, positions=[2],labels=['bas rep dif'])
    for xs,val in zip(xs_down_rep_dif, list_down_rep_dif):
        plt.scatter(xs, val, color='red')


    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_dif, positions=[3],labels=['haut rep dif'])
    for xs,val in zip(xs_up_rep_dif, list_up_rep_dif):
        plt.scatter(xs, val, color='green')

    plt.grid()
    plt.title("global LF reproduction en conditions identiques et différentes")
    plt.show()


    x = [1,2,3,4]
    m =[]
    for i in range (4):
        moyenne = (moyenne_question2[0][i] + moyenne_question2[1][i] + moyenne_question2[2][i] + moyenne_question2[3][i])/4
        m.append(moyenne)
    error_down_rep_id = np.std(m_down_rep_id, ddof=1) / np.sqrt(len(m_down_rep_id))
    error_up_rep_id = np.std(m_up_rep_id, ddof=1) / np.sqrt(len(m_up_rep_id))
    error_down_rep_dif = np.std(m_down_rep_dif, ddof=1) / np.sqrt(len(m_down_rep_dif))
    error_up_rep_dif = np.std(m_up_rep_dif, ddof=1) / np.sqrt(len(m_up_rep_dif))

    plt.scatter(x, moyenne_question2[0], color='red', label='Victor')
    plt.scatter(x, moyenne_question2[1], color='blue', label='Lise')
    plt.scatter(x, moyenne_question2[2], color='green', label='Sophie')
    plt.scatter(x, moyenne_question2[3], color='orange', label='Hugo')
    plt.scatter(x, m, color='black', label='Moyenne totale')
    plt.errorbar(1, m[0], yerr=error_down_rep_id, fmt='o', capsize=5, color='black', label='Erreur standard')
    plt.errorbar(2, m[1], yerr=error_up_rep_id, fmt='o', capsize=5, color='black')
    plt.errorbar(3, m[2], yerr=error_down_rep_dif, fmt='o', capsize=5, color='black')
    plt.errorbar(4, m[3], yerr=error_up_rep_dif, fmt='o', capsize=5, color='black')
    plt.legend()
    plt.ylim(3, 9)
    plt.ylabel('LF [N]')
    plt.xticks([1, 2, 3, 4], ['bas cond ident', 'haut cond ident', 'bas cond diff', 'haut cond diff'])
    plt.title("Moyenne LF reproduction en conditions identiques et différentes")
    plt.show()


    # Créer un DataFrame avec les données
    datatest = pd.DataFrame({
        'Force': m_down_rep_id + m_up_rep_id + m_down_rep_dif + m_up_rep_dif,  
        'Participant': nom * 4,
        'Direction': ['down'] * 4 + ['up'] * 4 + ['down'] * 4 + ['up'] * 4,
        'Condition': ['id'] * 8  + ['dif'] * 8
    })

    # Convertir les variables 'Direction', 'Gants' et 'Participant' en variables catégoriques
    datatest['Direction'] = pd.Categorical(datatest['Direction'])
    datatest['Condition'] = pd.Categorical(datatest['Condition'])
    datatest['Participant'] = pd.Categorical(datatest['Participant'])

    # Effectuer l'ANOVA à mesures répétées
    anova = AnovaRM(datatest, 'Force', 'Participant', within=['Direction', 'Condition'], aggregate_func='mean').fit()

    # Afficher les résultats de l'ANOVA à mesures répétées
    print(anova.summary())

elif partie == 'partie3':
    #comparer si reproduction lf egale quand entr avec/repr avec et entr sans/rep avec
    moyenne_question2 = []
    m_down_rep_id =[]
    m_up_rep_id =[]
    m_down_rep_dif =[]
    m_up_rep_dif =[]

    for n in nom :
        fig = plt.figure(figsize = [15,9])
        moyenne = []

        down_rep_id = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')&(df['gant']=='a')].iloc[:, 5:17:2] #down conditions identiques reproduction
        down_rep_id = down_rep_id.dropna()
        list_down_rep_id = down_rep_id.values.flatten()
        moyenne_down_rep_id = np.mean(list_down_rep_id)
        moyenne.append(moyenne_down_rep_id)
        m_down_rep_id.append(moyenne_down_rep_id)

        up_rep_id = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')&(df['gant']=='a')].iloc[:, 6:17:2] #up conditions identiques reproduction
        up_rep_id = up_rep_id.dropna()
        list_up_rep_id = up_rep_id.values.flatten()
        moyenne_up_rep_id = np.mean(list_up_rep_id)
        moyenne.append(moyenne_up_rep_id)
        m_up_rep_id.append(moyenne_up_rep_id)

        down_rep_dif = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')&(df['gant']=='a')].iloc[:, 5:17:2] #down conditions différentes reproduction
        down_rep_dif = down_rep_dif.dropna()
        list_down_rep_dif = down_rep_dif.values.flatten()
        moyenne_down_rep_dif = np.mean(list_down_rep_dif)
        moyenne.append(moyenne_down_rep_dif)
        m_down_rep_dif.append(moyenne_down_rep_dif)

        up_rep_dif = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')&(df['gant']=='a')].iloc[:, 6:17:2] #up conditions différentes reproduction
        up_rep_dif = up_rep_dif.dropna()
        list_up_rep_dif = up_rep_dif.values.flatten()
        moyenne_up_rep_dif = np.mean(list_up_rep_dif)
        moyenne.append(moyenne_up_rep_dif)
        m_up_rep_dif.append(moyenne_up_rep_dif)

        moyenne_question2.append(moyenne)

        xs_down_rep_id = []
        for i in range(len(list_down_rep_id)):
            xs_down_rep_id.append(np.random.normal((i+1)*10e-20,0.02))

        xs_up_rep_id = []
        for i in range(len(list_up_rep_id)):
            xs_up_rep_id.append(np.random.normal((i+1)*10e-20+1,0.02))

        xs_down_rep_dif = []
        for i in range(len(list_down_rep_dif)):
            xs_down_rep_dif.append(np.random.normal((i+1)*10e-20+2,0.02))

        xs_up_rep_dif= []
        for i in range(len(list_up_rep_dif)):
            xs_up_rep_dif.append(np.random.normal((i+1)*10e-20+3,0.02))


        #create box plots for the even indexes of the line
        plt.boxplot(list_down_rep_id, positions=[0],labels=['bas entr avec gant'])
        for xs,val in zip(xs_down_rep_id, list_down_rep_id):
            plt.scatter(xs, val, color='red')

        #create box plots for the odd indexes of the line
        plt.boxplot(list_up_rep_id, positions=[1],labels=['haut entr avec gant'])
        for xs,val in zip(xs_up_rep_id, list_up_rep_id):
            plt.scatter(xs, val, color='green')

        #create box plots for the even indexes of the line
        plt.boxplot(list_down_rep_dif, positions=[2],labels=['bas entr sans gant'])
        for xs,val in zip(xs_down_rep_dif, list_down_rep_dif):
            plt.scatter(xs, val, color='red')


        #create box plots for the odd indexes of the line
        plt.boxplot(list_up_rep_dif, positions=[3],labels=['haut entr sans gant'])
        for xs,val in zip(xs_up_rep_dif, list_up_rep_dif):
            plt.scatter(xs, val, color='green')

        plt.grid()
        plt.title(n +" LF rep avec gant conditions d'entrainement identiques et différentes")
        plt.show()

    ##global##

    fig = plt.figure(figsize = [15,9])

    down_rep_id = df[( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')&(df['gant']=='a')].iloc[:, 5:17:2] #down conditions identiques reproduction
    down_rep_id = down_rep_id.dropna()
    list_down_rep_id = down_rep_id.values.flatten()

    up_rep_id = df[(df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')&(df['gant']=='a')].iloc[:, 6:17:2] #up conditions identiques reproduction
    up_rep_id = up_rep_id.dropna()
    list_up_rep_id = up_rep_id.values.flatten()

    down_rep_dif = df[(df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')&(df['gant']=='a')].iloc[:, 5:17:2] #down conditions différentes reproduction
    down_rep_dif = down_rep_dif.dropna()
    list_down_rep_dif = down_rep_dif.values.flatten()

    up_rep_dif = df[(df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')&(df['gant']=='a')].iloc[:, 6:17:2] #up conditions différentes reproduction
    up_rep_dif = up_rep_dif.dropna()
    list_up_rep_dif = up_rep_dif.values.flatten()

    xs_down_rep_id = []
    for i in range(len(list_down_rep_id)):
        xs_down_rep_id.append(np.random.normal((i+1)*10e-20,0.02))

    xs_up_rep_id = []
    for i in range(len(list_up_rep_id)):
        xs_up_rep_id.append(np.random.normal((i+1)*10e-20+1,0.02))

    xs_down_rep_dif = []
    for i in range(len(list_down_rep_dif)):
        xs_down_rep_dif.append(np.random.normal((i+1)*10e-20+2,0.02))

    xs_up_rep_dif= []
    for i in range(len(list_up_rep_dif)):
        xs_up_rep_dif.append(np.random.normal((i+1)*10e-20+3,0.02))


    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_id, positions=[0],labels=['bas entr avec gant'])
    for xs,val in zip(xs_down_rep_id, list_down_rep_id):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_id, positions=[2],labels=['haut entr avec gant'])
    for xs,val in zip(xs_up_rep_id, list_up_rep_id):
        plt.scatter(xs, val, color='green')

    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_dif, positions=[1],labels=['bas entr sans gant'])
    for xs,val in zip(xs_down_rep_dif, list_down_rep_dif):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_dif, positions=[3],labels=['haut entr sans gant'])
    for xs,val in zip(xs_up_rep_dif, list_up_rep_dif):
        plt.scatter(xs, val, color='green')

    plt.grid()
    plt.title("global LF reproduction avec gant en conditions d'entrainement identiques et différentes")
    plt.show()


    x = [1,2,3,4]
    m =[]
    for i in range (4):
        moyenne = (moyenne_question2[0][i] + moyenne_question2[1][i] + moyenne_question2[2][i] + moyenne_question2[3][i])/4
        m.append(moyenne)
    error_down_rep_id = np.std(m_down_rep_id, ddof=1) / np.sqrt(len(m_down_rep_id))
    error_up_rep_id = np.std(m_up_rep_id, ddof=1) / np.sqrt(len(m_up_rep_id))
    error_down_rep_dif = np.std(m_down_rep_dif, ddof=1) / np.sqrt(len(m_down_rep_dif))
    error_up_rep_dif = np.std(m_up_rep_dif, ddof=1) / np.sqrt(len(m_up_rep_dif))

    plt.scatter(x, moyenne_question2[0], color='red', label='Victor')
    plt.scatter(x, moyenne_question2[1], color='blue', label='Lise')
    plt.scatter(x, moyenne_question2[2], color='green', label='Sophie')
    plt.scatter(x, moyenne_question2[3], color='orange', label='Hugo')
    plt.scatter(x, m, color='black', label='Moyenne totale')
    plt.errorbar(1, m[0], yerr=error_down_rep_id, fmt='o', capsize=5, color='black', label='Erreur standard')
    plt.errorbar(2, m[1], yerr=error_up_rep_id, fmt='o', capsize=5, color='black')
    plt.errorbar(3, m[2], yerr=error_down_rep_dif, fmt='o', capsize=5, color='black')
    plt.errorbar(4, m[3], yerr=error_up_rep_dif, fmt='o', capsize=5, color='black')
    plt.legend()
    plt.ylim(3, 9)
    plt.ylabel('LF [N]')
    plt.xticks([1, 2, 3, 4], ['bas entr avec gant', 'haut entr avec gant', 'bas entr sans gant', 'haut entr sans gant'])
    plt.title("Moyenne LF reproduction avec gant")
    plt.show()


    # Créer un DataFrame avec les données
    datatest = pd.DataFrame({
        'Force': m_down_rep_id + m_up_rep_id + m_down_rep_dif + m_up_rep_dif,  
        'Participant': nom * 4,
        'Direction': ['down'] * 4 + ['up'] * 4 + ['down'] * 4 + ['up'] * 4,
        'Condition': ['id'] * 8  + ['dif'] * 8
    })

    # Convertir les variables 'Direction', 'Gants' et 'Participant' en variables catégoriques
    datatest['Direction'] = pd.Categorical(datatest['Direction'])
    datatest['Condition'] = pd.Categorical(datatest['Condition'])
    datatest['Participant'] = pd.Categorical(datatest['Participant'])

    # Effectuer l'ANOVA à mesures répétées
    anova = AnovaRM(datatest, 'Force', 'Participant', within=['Direction', 'Condition'], aggregate_func='mean').fit()

    # Afficher les résultats de l'ANOVA à mesures répétées
    print(anova.summary())

elif partie == 'partie4':
    #comparer si reproduction lf egale quand entr sans/repr sans et entr avec/rep sans
    moyenne_question2 = []
    m_down_rep_id =[]
    m_up_rep_id =[]
    m_down_rep_dif =[]
    m_up_rep_dif =[]

    for n in nom :
        fig = plt.figure(figsize = [15,9])
        moyenne = []

        down_rep_id = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')&(df['gant']=='s')].iloc[:, 5:17:2] #down conditions identiques reproduction
        down_rep_id = down_rep_id.dropna()
        list_down_rep_id = down_rep_id.values.flatten()
        moyenne_down_rep_id = np.mean(list_down_rep_id)
        moyenne.append(moyenne_down_rep_id)
        m_down_rep_id.append(moyenne_down_rep_id)

        up_rep_id = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')&(df['gant']=='s')].iloc[:, 6:17:2] #up conditions identiques reproduction
        up_rep_id = up_rep_id.dropna()
        list_up_rep_id = up_rep_id.values.flatten()
        moyenne_up_rep_id = np.mean(list_up_rep_id)
        moyenne.append(moyenne_up_rep_id)
        m_up_rep_id.append(moyenne_up_rep_id)

        down_rep_dif = df[(df['nom'] == n )&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')&(df['gant']=='s')].iloc[:, 5:17:2] #down conditions différentes reproduction
        down_rep_dif = down_rep_dif.dropna()
        list_down_rep_dif = down_rep_dif.values.flatten()
        moyenne_down_rep_dif = np.mean(list_down_rep_dif)
        moyenne.append(moyenne_down_rep_dif)
        m_down_rep_dif.append(moyenne_down_rep_dif)

        up_rep_dif = df[(df['nom'] == n)&( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')&(df['gant']=='s')].iloc[:, 6:17:2] #up conditions différentes reproduction
        up_rep_dif = up_rep_dif.dropna()
        list_up_rep_dif = up_rep_dif.values.flatten()
        moyenne_up_rep_dif = np.mean(list_up_rep_dif)
        moyenne.append(moyenne_up_rep_dif)
        m_up_rep_dif.append(moyenne_up_rep_dif)

        moyenne_question2.append(moyenne)

        xs_down_rep_id = []
        for i in range(len(list_down_rep_id)):
            xs_down_rep_id.append(np.random.normal((i+1)*10e-20,0.02))

        xs_up_rep_id = []
        for i in range(len(list_up_rep_id)):
            xs_up_rep_id.append(np.random.normal((i+1)*10e-20+1,0.02))

        xs_down_rep_dif = []
        for i in range(len(list_down_rep_dif)):
            xs_down_rep_dif.append(np.random.normal((i+1)*10e-20+2,0.02))

        xs_up_rep_dif= []
        for i in range(len(list_up_rep_dif)):
            xs_up_rep_dif.append(np.random.normal((i+1)*10e-20+3,0.02))


        #create box plots for the even indexes of the line
        plt.boxplot(list_down_rep_id, positions=[0],labels=['bas entr sans gant'])
        for xs,val in zip(xs_down_rep_id, list_down_rep_id):
            plt.scatter(xs, val, color='red')

        #create box plots for the odd indexes of the line
        plt.boxplot(list_up_rep_id, positions=[1],labels=['haut entr sans gant'])
        for xs,val in zip(xs_up_rep_id, list_up_rep_id):
            plt.scatter(xs, val, color='green')

        #create box plots for the even indexes of the line
        plt.boxplot(list_down_rep_dif, positions=[2],labels=['bas entr avec gant'])
        for xs,val in zip(xs_down_rep_dif, list_down_rep_dif):
            plt.scatter(xs, val, color='red')


        #create box plots for the odd indexes of the line
        plt.boxplot(list_up_rep_dif, positions=[3],labels=['haut entr avec gant'])
        for xs,val in zip(xs_up_rep_dif, list_up_rep_dif):
            plt.scatter(xs, val, color='green')

        plt.grid()
        plt.title(n +" LF rep sans gant conditions identiques et différentes")
        plt.show()

    ##global##

    fig = plt.figure(figsize = [15,9])

    down_rep_id = df[( df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')&(df['gant']=='s')].iloc[:, 5:17:2] #down conditions identiques reproduction
    down_rep_id = down_rep_id.dropna()
    list_down_rep_id = down_rep_id.values.flatten()

    up_rep_id = df[(df['type']=='rep')&(df['force']=='lf')&(df['cond']=='id')&(df['gant']=='s')].iloc[:, 6:17:2] #up conditions identiques reproduction
    up_rep_id = up_rep_id.dropna()
    list_up_rep_id = up_rep_id.values.flatten()

    down_rep_dif = df[(df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')&(df['gant']=='s')].iloc[:, 5:17:2] #down conditions différentes reproduction
    down_rep_dif = down_rep_dif.dropna()
    list_down_rep_dif = down_rep_dif.values.flatten()

    up_rep_dif = df[(df['type']=='rep')&(df['force']=='lf')&(df['cond']=='dif')&(df['gant']=='s')].iloc[:, 6:17:2] #up conditions différentes reproduction
    up_rep_dif = up_rep_dif.dropna()
    list_up_rep_dif = up_rep_dif.values.flatten()

    xs_down_rep_id = []
    for i in range(len(list_down_rep_id)):
        xs_down_rep_id.append(np.random.normal((i+1)*10e-20,0.02))

    xs_up_rep_id = []
    for i in range(len(list_up_rep_id)):
        xs_up_rep_id.append(np.random.normal((i+1)*10e-20+1,0.02))

    xs_down_rep_dif = []
    for i in range(len(list_down_rep_dif)):
        xs_down_rep_dif.append(np.random.normal((i+1)*10e-20+2,0.02))

    xs_up_rep_dif= []
    for i in range(len(list_up_rep_dif)):
        xs_up_rep_dif.append(np.random.normal((i+1)*10e-20+3,0.02))


    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_id, positions=[0],labels=['bas entr sans gant'])
    for xs,val in zip(xs_down_rep_id, list_down_rep_id):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_id, positions=[2],labels=['haut entr sans gant'])
    for xs,val in zip(xs_up_rep_id, list_up_rep_id):
        plt.scatter(xs, val, color='green')

    #create box plots for the even indexes of the line
    plt.boxplot(list_down_rep_dif, positions=[1],labels=['bas entr avec gant'])
    for xs,val in zip(xs_down_rep_dif, list_down_rep_dif):
        plt.scatter(xs, val, color='red')

    #create box plots for the odd indexes of the line
    plt.boxplot(list_up_rep_dif, positions=[3],labels=['haut entr avec gant'])
    for xs,val in zip(xs_up_rep_dif, list_up_rep_dif):
        plt.scatter(xs, val, color='green')

    plt.grid()
    plt.title("global LF reproduction en conditions identiques et différentes")
    plt.show()


    x = [1,2,3,4]
    m =[]
    for i in range (4):
        moyenne = (moyenne_question2[0][i] + moyenne_question2[1][i] + moyenne_question2[2][i] + moyenne_question2[3][i])/4
        m.append(moyenne)
    error_down_rep_id = np.std(m_down_rep_id, ddof=1) / np.sqrt(len(m_down_rep_id))
    error_up_rep_id = np.std(m_up_rep_id, ddof=1) / np.sqrt(len(m_up_rep_id))
    error_down_rep_dif = np.std(m_down_rep_dif, ddof=1) / np.sqrt(len(m_down_rep_dif))
    error_up_rep_dif = np.std(m_up_rep_dif, ddof=1) / np.sqrt(len(m_up_rep_dif))

    plt.scatter(x, moyenne_question2[0], color='red', label='Victor')
    plt.scatter(x, moyenne_question2[1], color='blue', label='Lise')
    plt.scatter(x, moyenne_question2[2], color='green', label='Sophie')
    plt.scatter(x, moyenne_question2[3], color='orange', label='Hugo')
    plt.scatter(x, m, color='black', label='Moyenne totale')
    plt.errorbar(1, m[0], yerr=error_down_rep_id, fmt='o', capsize=5, color='black', label='Erreur standard')
    plt.errorbar(2, m[1], yerr=error_up_rep_id, fmt='o', capsize=5, color='black')
    plt.errorbar(3, m[2], yerr=error_down_rep_dif, fmt='o', capsize=5, color='black')
    plt.errorbar(4, m[3], yerr=error_up_rep_dif, fmt='o', capsize=5, color='black')
    plt.legend()
    plt.ylim(3, 9)
    plt.ylabel('LF [N]')
    plt.xticks([1, 2, 3, 4], ['bas entr sans gant', 'haut entr sans gant', 'bas entr avec gant', 'haut entr avec gant'])
    plt.title("Moyenne LF reproduction sans gant")
    plt.show()


    # Créer un DataFrame avec les données
    datatest = pd.DataFrame({
        'Force': m_down_rep_id + m_up_rep_id + m_down_rep_dif + m_up_rep_dif,  
        'Participant': nom * 4,
        'Direction': ['down'] * 4 + ['up'] * 4 + ['down'] * 4 + ['up'] * 4,
        'Condition': ['id'] * 8  + ['dif'] * 8
    })

    # Convertir les variables 'Direction', 'Gants' et 'Participant' en variables catégoriques
    datatest['Direction'] = pd.Categorical(datatest['Direction'])
    datatest['Condition'] = pd.Categorical(datatest['Condition'])
    datatest['Participant'] = pd.Categorical(datatest['Participant'])

    # Effectuer l'ANOVA à mesures répétées
    anova = AnovaRM(datatest, 'Force', 'Participant', within=['Direction', 'Condition'], aggregate_func='mean').fit()

    # Afficher les résultats de l'ANOVA à mesures répétées
    print(anova.summary())

    from scipy.stats import ttest_ind
    sample1 = m_down_rep_id + m_down_rep_dif #direction
    sample2 = m_up_rep_id + m_up_rep_dif    #direction
    sample3 = m_down_rep_id + m_up_rep_id #condition
    sample4 = m_down_rep_dif + m_up_rep_dif #condition

    t_statistic, p_value = ttest_ind(sample1, sample2)

    # Afficher les résultats
    print("Test statistic:", t_statistic)
    print("p-value:", p_value)


elif partie == 'partie5':

    #comparer les marges de sécurité de la grip force avec et sans gants (et différencier up et down)
    #LF<=GF*µ -> on va regarder µ*GF-LF qui est égal à la marge de sécurité 


    moyenne_question3 = []
    m_marge_down_a =[]
    m_marge_up_a =[]
    m_marge_down_s =[]
    m_marge_up_s =[]

    # µ = k(GF)^n
    k_list_sans = [ 1.54,1.446,2.07,1.14]  #droit seulement
    n_list_sans = [-0.234,-0.215,-0.12,-0.188]  #droit seulement

    #k_list_sans = [ 1.66,1.37,1.88,1.054]  #moyenne droite et gauche
    #n_list_sans = [-0.045,-0.23, -0.015, -0.148]   #moyenne droite et gauche

    #mu_list_avec = [0.745,1.28,2.29,0.99]  #seuil à 3  #droit et gauche
    mu_list_avec = [0.63,1.18,2.297,0.98]    #seuil à 4  #droit et gauche
    #mu_list_avec = [0.59,1.13,2.46,0.96]    #seuil à 4  #juste droit



    for n in nom :
        k_sans = 0
        n_sans = 0
        mu_avec = 0

        if n == 'Victor':
            k_sans = k_list_sans[0]
            n_sans = n_list_sans[0]
            mu_avec = mu_list_avec[0]

        elif n == 'Lise':
            k_sans = k_list_sans[1]
            n_sans = n_list_sans[1]
            mu_avec = mu_list_avec[1]

        elif n == 'Sophie':
            k_sans = k_list_sans[2]
            n_sans = n_list_sans[2]
            mu_avec = mu_list_avec[2]

        else :
            k_sans = k_list_sans[3]
            n_sans = n_list_sans[3]
            mu_avec = mu_list_avec[3]

        fig = plt.figure(figsize = [15,9])
        moyenne = []


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

        #calcul des marges de sécurité
        marge_down_a = []
        for i in range(len(list_down_rep_a_lf)):
            marge_down_a.append(list_down_rep_a_gf[i]*mu_avec-list_down_rep_a_lf[i])
        moyenne_marge_down_a = np.mean(marge_down_a)
        moyenne.append(moyenne_marge_down_a)
        m_marge_down_a.append(moyenne_marge_down_a)

        marge_up_a = []
        for i in range(len(list_up_rep_a_lf)):
            marge_up_a.append(list_up_rep_a_gf[i]*mu_avec-list_up_rep_a_lf[i])
        moyenne_marge_up_a = np.mean(marge_up_a)
        moyenne.append(moyenne_marge_up_a)
        m_marge_up_a.append(moyenne_marge_up_a)

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

        #calcul des marges de sécurité
        marge_down_s = []
        for i in range(len(list_down_rep_s_lf)):
            mu_sans = k_sans * (list_down_rep_s_gf[i]**n_sans)
            marge_down_s.append(list_down_rep_s_gf[i]*mu_sans-list_down_rep_s_lf[i])
        moyenne_marge_down_s = np.mean(marge_down_s)
        moyenne.append(moyenne_marge_down_s)
        m_marge_down_s.append(moyenne_marge_down_s)

        marge_up_s = []
        for i in range(len(list_up_rep_s_lf)):
            mu_sans = k_sans * (list_up_rep_s_gf[i]**n_sans)
            marge_up_s.append(list_up_rep_s_gf[i]*mu_sans-list_up_rep_s_lf[i])
        moyenne_marge_up_s = np.mean(marge_up_s)
        moyenne.append(moyenne_marge_up_s)
        m_marge_up_s.append(moyenne_marge_up_s)


        moyenne_question3.append(moyenne)

        xs_down_a = []
        for i in range(len(marge_down_a)):
            xs_down_a.append(np.random.normal((i+1)*10e-20,0.02))

        xs_up_a = []
        for i in range(len(marge_up_a)):
            xs_up_a.append(np.random.normal((i+1)*10e-20+1,0.02))

        xs_down_s = []
        for i in range(len(marge_down_s)):
            xs_down_s.append(np.random.normal((i+1)*10e-20+2,0.02))

        xs_up_s= []
        for i in range(len(marge_up_s)):
            xs_up_s.append(np.random.normal((i+1)*10e-20+3,0.02))


        #create box plots for the even indexes of the line
        plt.boxplot(marge_down_a, positions=[0],labels=['down avec gant'])
        for xs,val in zip(xs_down_a, marge_down_a):
            plt.scatter(xs, val, color='red')

        #create box plots for the even indexes of the line
        plt.boxplot(marge_down_s, positions=[2],labels=['down sans gant'])
        for xs,val in zip(xs_down_s, marge_down_s):
            plt.scatter(xs, val, color='red')

        #create box plots for the odd indexes of the line
        plt.boxplot(marge_up_a, positions=[1],labels=['up avec gant'])
        for xs,val in zip(xs_up_a, marge_up_a):
            plt.scatter(xs, val, color='green')

        #create box plots for the odd indexes of the line
        plt.boxplot(marge_up_s, positions=[3],labels=['up sans gant'])
        for xs,val in zip(xs_up_s, marge_up_s):
            plt.scatter(xs, val, color='green')

        plt.grid()
        plt.title(n +" marges de sécurité avec et sans gants")
        plt.show()

    x = [1,2,3,4]
    m =[]
    for i in range (4):
        moyenne = (moyenne_question3[0][i] + moyenne_question3[1][i] + moyenne_question3[2][i] + moyenne_question3[3][i])/4
        m.append(moyenne)
    error_marge_down_a = np.std(m_marge_down_a, ddof=1) / np.sqrt(len(m_marge_down_a))
    error_marge_up_a = np.std(m_marge_up_a , ddof=1) / np.sqrt(len(m_marge_up_a ))
    error_marge_down_s = np.std(m_marge_down_s, ddof=1) / np.sqrt(len(m_marge_down_s))
    error_marge_up_s = np.std(m_marge_up_s, ddof=1) / np.sqrt(len(m_marge_up_s))

    plt.scatter(x, moyenne_question3[0], color='red', label='Victor')
    plt.scatter(x, moyenne_question3[1], color='blue', label='Lise')
    plt.scatter(x, moyenne_question3[2], color='green', label='Sophie')
    plt.scatter(x, moyenne_question3[3], color='orange', label='Hugo')
    plt.scatter(x, m, color='black', label='Moyenne totale')
    plt.errorbar(1, m[0], yerr=error_marge_down_a , fmt='o', capsize=5, color='black', label='Erreur standard')
    plt.errorbar(2, m[1], yerr=error_marge_up_a, fmt='o', capsize=5, color='black')
    plt.errorbar(3, m[2], yerr=error_marge_down_s, fmt='o', capsize=5, color='black')
    plt.errorbar(4, m[3], yerr=error_marge_up_s, fmt='o', capsize=5, color='black')
    plt.legend(fontsize='small')
    plt.ylabel('Marge')
    plt.xticks([1, 2, 3, 4], ['bas avec gant', 'haut avec gant', 'bas sans gant', 'haut sans gant'])
    plt.title("Marge de sécurité de GF avec et sans gant")
    plt.show()


    # Créer un DataFrame avec les données
    datatest = pd.DataFrame({
        'Marge': m_marge_down_a + m_marge_up_a  + m_marge_down_s  + m_marge_up_s,  
        'Participant': nom * 4,
        'Direction': ['down'] * 4 + ['up'] * 4 + ['down'] * 4 + ['up'] * 4,
        'Gant': ['avec'] * 8  + ['sans'] * 8
    })

    # Convertir les variables 'Direction', 'Gants' et 'Participant' en variables catégoriques
    datatest['Direction'] = pd.Categorical(datatest['Direction'])
    datatest['Gant'] = pd.Categorical(datatest['Gant'])
    datatest['Participant'] = pd.Categorical(datatest['Participant'])

    # Effectuer l'ANOVA à mesures répétées
    anova = AnovaRM(datatest, 'Marge', 'Participant', within=['Direction', 'Gant'], aggregate_func='mean').fit()

    # Afficher les résultats de l'ANOVA à mesures répétées
    print(anova.summary())

    from scipy.stats import ttest_ind
    sample1 = m_marge_down_a + m_marge_down_s  #direction
    sample2 = m_marge_up_a + m_marge_up_s
    sample3 = m_marge_down_a + m_marge_up_a  #gant
    sample4 = m_marge_down_s + m_marge_up_s

    t_statistic, p_value = ttest_ind(sample3, sample4)

    # Afficher les résultats
    print("Test statistic:", t_statistic)
    print("p-value:", p_value)
