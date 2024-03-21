# code pour ecrire les listes dans un fichier texte : une liste par ligne

# Deux listes à enregistrer dans le fichier
liste1 = [1.5,2,3]
liste2 = [4,5,6]

# Chemin du fichier dans lequel vous souhaitez enregistrer les listes
chemin_fichier = "test.txt"

# Ouverture du fichier en mode écriture
with open(chemin_fichier, 'a') as fichier:
    # Écriture de la première liste
    ligne1 = ' '.join(map(str, liste1))
    
    # Écriture de la ligne dans le fichier
    fichier.write(ligne1)

    fichier.write("\n")
    
    # Écriture de la deuxième liste
    ligne2 = ' '.join(map(str, liste2))
    
    # Écriture de la ligne dans le fichier
    fichier.write(ligne2)
    fichier.write("\n")


print("Deux listes enregistrées avec succès dans le fichier", chemin_fichier)