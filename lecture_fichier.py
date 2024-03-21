#code pour lire le fichier et récupérer les listes (retransformation en float!!)

# Chemin du fichier dans lequel vous avez enregistré les listes
chemin_fichier = "test.txt"

# Listes pour stocker les éléments récupérés
liste1 = []
liste2 = []

# Lecture du contenu du fichier
with open(chemin_fichier, 'r') as fichier:
    lignes = fichier.readlines()
    
    # Récupération des éléments de la première liste
    liste1 = lignes[0].strip().split()
    
    # Récupération des éléments de la deuxième liste
    liste2 = lignes[1].strip().split()
for i in range (len(liste1)):
    liste1[i] = float(liste1[i])
for i in range (len(liste2)):
    liste2[i] = float(liste2[i])
print("Liste 1:", liste1)
print("Liste 2:", liste2)
