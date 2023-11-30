# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt
#from mtcnn import MTCNN


#Ceci est une ebauche pour la localisation de visage

# reading an image
im_obj = cv2.imread("temp/06.jpg")

def crop(im):
    # Redimensionnez l'image à la nouvelle taille
    nouvelle_longueur = 300
    ratio=nouvelle_longueur/im.shape[1]
    nouvelle_largeur=int(im.shape[0]*ratio)
    image_redimensionnee = cv2.resize(im, (nouvelle_longueur,nouvelle_largeur))
    
    im = image_redimensionnee
    # Charger le classificateur de visages pré-entrainé
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    try:
        # Détecter les visages dans l'image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if not(faces):
            raise Exception("Une erreur s'est produite : pas de détection de visage", 500)
    
    except Exception as e:
        # Gérer l'exception
        print(f"Erreur : {e}")
    
    # Parcourir la liste des visages détectés et les afficher
    for (x, y, w, h) in faces:
        # Dessiner un rectangle autour du visage
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
        # Récupérer la région d'intérêt (ROI) du visage
        face_roi = gray[y:y + h, x:x + w]
    
        # Afficher le visage
        cv2.imshow('Face', face_roi)
        cv2.waitKey(0)  # Attendre une touche pour passer au visage suivant
        cv2.destroyAllWindows()
    return faces




