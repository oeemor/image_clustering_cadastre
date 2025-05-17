import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import sys
import math

k = 10 
show = False

for i in range(len(sys.argv)):
    try:
        if sys.argv[i] == '--dir':
            image_folder = sys.argv[i + 1]
            
    except IndexError:
        print("Usage: python histo.py --dir <image_folder>")
        sys.exit(1)
    try:
        if sys.argv[i] in ['-n', '--nb-clusters']:
            k = int(sys.argv[i + 1])
            
    except IndexError:
        print("Usage: python histo.py --nb-clusters <k>")
        sys.exit(1)
    
    if sys.argv[i] == '--show':
        show = True
        
    if sys.argv[i] in ['-h', '--help']:
        print("Usage: python histo.py --dir <image_folder> --nb-clusters <k>")
        print("Options:")
        print("  --dir <image_folder> : Dossier contenant les images à traiter.")
        print("  --nb-clusters <k> : Nombre de clusters pour le KMeans.")
        print("  --show : Afficher les images des clusters.")
        sys.exit(0)
    
if not os.path.exists(image_folder) or not os.path.isdir(image_folder):
    print(f"Le dossier {image_folder} n'existe pas/ou n'est pas un dossier valide.")
    print("Usage: python histo.py --dir <image_folder>")
    sys.exit(1)


def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


features = []
filenames = []

for file in os.listdir(image_folder):
    if file.endswith('.jpg') or file.endswith('.png'):
        path = os.path.join(image_folder, file)
        image = cv2.imread(path)
        hist = extract_color_histogram(image)
        features.append(hist)
        filenames.append(file)

features = np.array(features)

kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(features)

for i in range(k):
    print(f"Cluster {i+1}:")
    cluster_files = [filenames[j] for j in range(len(filenames)) if labels[j] == i]
    print(cluster_files)

def afficher_images_cluster(cluster_num, filenames, labels, image_folder):
    images_cluster = [os.path.join(image_folder, f) for i, f in enumerate(filenames) if labels[i] == cluster_num] 
    plt.figure(figsize=(10, 10))
    for idx, img_path in enumerate(images_cluster):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        grid_size = math.ceil(math.sqrt(len(images_cluster)))#je prend la racine carré de la taille du cluster arondie au sup pour avoir une grille carré de la bonne taille
        plt.subplot(grid_size, grid_size, idx+1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(f"Images du Cluster {cluster_num+1}")
    plt.savefig(f"Cluster {i+1}.png")

for i in range(k):
    afficher_images_cluster(i, filenames, labels, image_folder)