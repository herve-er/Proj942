# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:09:36 2023

@author: yovodevz
"""
import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import shutil


def redim(image,nouvelle_taille):
# Vérifiez si l'image est correctement chargée
    ratio = nouvelle_taille / image.shape[1]
    nouvelle_largeur = int(image.shape[0] * ratio)
    image_redimensionnee = cv2.resize(image, (nouvelle_taille, nouvelle_largeur))
    return(image_redimensionnee) 
    

# Fonctions
def read_images(path, nouvelle_longueur=None):
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        dirnames.sort()
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given )
                    # resize to given size (if given)
                    if nouvelle_longueur is not None:
                        ratio = nouvelle_longueur / im.width
                        nouvelle_largeur = int(im.height * ratio)
                        im = im.resize((nouvelle_longueur, nouvelle_largeur), Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as e:
                    print(f"Erreur d'E/S ({e.errno}): {e.strerror}")
                except Exception as e:
                    print(f"Erreur inattendue : {str(e)}")
                    raise
            c = c + 1
    return [X, y]


def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1, -1)))
    return mat


def asColumnMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
    for col in X:
        mat = np.hstack((mat, np.asarray(col).reshape(-1, 1)))
    return mat


def pca(X, y, num_components=0):
    [n, d] = X.shape
    if (num_components <= 0) or (num_components > n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n > d:
        C = np.dot(X.T, X)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    else:
        C = np.dot(X, X.T)
        [eigenvalues, eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T, eigenvectors)
        for i in range(n):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
    # or simply perform an economy size decomposition
    # eigenvectors , eigenvalues , variance = np.linalg.svd (X.T, full_matrices = False )
    # sort eigenvectors descending by their eigenvalue
    idx = np.argsort(- eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # select only num_components
    eigenvalues = eigenvalues[0: num_components].copy()
    eigenvectors = eigenvectors[:, 0: num_components].copy()
    return [eigenvalues, eigenvectors, mu]


def project(W, X, mu=None):
    if mu is None:
        return np.dot(X, W)
    return np.dot(X - mu, W)


def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu


def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [ low...high ].
    X = X * (high - low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


def create_font(fontname='Tahoma', fontsize=10):
    return {'fontname': fontname, 'fontsize': fontsize}


def subplot(title, images, rows, cols, sptitle=" subplot ", sptitles=[], colormap=cm.gray, ticks_visible=True,
            filename=None):
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center')
    for i in range(len(images)):
        ax0 = fig.add_subplot(rows, cols, (i + 1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma', 10))
        else:
            plt.title("%s #%d" % (sptitle, (i + 1)), create_font('Tahoma', 10))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Classes

class AbstractDistance(object):

    def __init__(self, name):
        self._name = name

    def __call__(self, p, q):
        raise NotImplementedError(" Every AbstractDistance must implement the __call__method.")

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return self._name


class EuclideanDistance(AbstractDistance):
    def __init__(self):
        AbstractDistance.__init__(self, " EuclideanDistance ")

    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return np.sqrt(np.sum(np.power((p - q), 2)))


class CosineDistance(AbstractDistance):
    def __init__(self):
        AbstractDistance.__init__(self, " CosineDistance ")

    def __call__(self, p, q):
        p = np.asarray(p).flatten()
        q = np.asarray(q).flatten()
        return -np.dot(p.T, q) / (np.sqrt(np.dot(p, p.T) * np.dot(q, q.T)))


class BaseModel(object):
    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):
        self.dist_metric = dist_metric
        self.num_components = 0
        self.projections = []
        self.W = []
        self.mu = []
        if (X is not None) and (y is not None):
            self.compute(X, y)

    def compute(self, X, y):
        raise NotImplementedError(" Every BaseModel must implement the compute method.")

    def predict(self, X):
        minDist = np.finfo('float').max
        minClass = -1
        
        Q = project(self.W, X.reshape(1, -1), self.mu)

        for i in range(len(self.projections)):
            dist = self.dist_metric(self.projections[i], Q)
            if dist < minDist:
                minDist = dist
                minClass = y[i]

        predicted_name = num_to_name.get(minClass, "Inconnu")
        
        return predicted_name


class EigenfacesModel(BaseModel):
    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):
        super(EigenfacesModel, self).__init__(X=X, y=y, dist_metric=dist_metric, num_components=num_components)

    def compute(self, X, y):
        [D, self.W, self.mu] = pca(asRowMatrix(X), y, self.num_components)
        # store labels
        self.y = y
        # store projections
        for xi in X:
            self.projections.append(project(self.W, xi.reshape(1, -1), self.mu))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

def predict(path,X,y,nouvelle_taille):
    # Chargez le classificateur Haar Cascade pré-entraîné pour la détection faciale
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Chargez l'image
    im = cv2.imread(path)
    im = redim(im, nouvelle_taille)
    predictions=[]
    # Convertissez l'image en niveaux de gris
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    try:
        # Détectez les visages dans l'image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces)==0:
            raise Exception("Aucun visage n'a été détecté dans l'image.")
        # Enregistrez chaque visage détecté dans le dossier des résultats des visages
        for i, (x, y, w, h) in enumerate(faces):
            # Récupérer la région d'intérêt (ROI) du visage
            face_roi = gray[y:y + h, x:x + w]
            
            # Créez un nouveau nom de fichier avec le suffixe du visage
            nom_fichier_visage = f'{os.path.splitext(os.path.basename(path))[0]}.jpg'
            
            # Afficher le visage
            '''cv2.imshow(f"Visage {i}", face_roi)
            cv2.waitKey(0)

            cv2.destroyAllWindows()'''

            # Demander à l'utilisateur si le visage est correct
          
            chemin_visage_crop = os.path.join(dossier_resultats, nom_fichier_visage)
            cv2.imwrite(chemin_visage_crop, face_roi)
            imtest = Image.open(chemin_visage_crop)
            imtest = imtest.convert("L")
            test = np.asarray(imtest, dtype=np.uint8)
            test2=redim(test,nouvelle_taille)
            # model computation
            model = EigenfacesModel(X, y)
            prediction=model.predict(test2)
            predictions.append(prediction)
        return(predictions)
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

#def ajout_a_la_base(prediction,chemin_visage_crop):
#    reponse_utilisateur_prediction = input('La prediction est correct ? (Oui/Non): ')
#    if reponse_utilisateur_prediction.lower() == 'oui':
#        #enregistrer l'image dans la base dans le fichier correspondant
#        nom_personne=prediction
#    else:
#        #demander quelle est le nom et la personne et l'enregister dans le bon fichier
#        nom_personne = input("Entrez le nom de la personne : ")
#    nom_personne = str(nom_personne).capitalize()
#    destination = os.path.join('base_sni', nom_personne)
#    
#    if not os.path.exists(destination):
#        os.makedirs(destination)
#    shutil.copy(chemin_visage_crop, destination)
#    
#    # Enregistrez l'image dans le chemin complet
#
#

# importer la base d'image
def init(path_base_image):
    sys.path.append("")
    [X, y] = read_images(path_base_image)
    # Preprocessing image on réduit la taille de l'image à 300
    new_X = []
    for image_i in X:
        # Redimensionnez l'image à la nouvelle taille
        image_redimensionnee=redim(image_i,300)
        new_X.append(image_redimensionnee)

    X = new_X
    # perform a full pca
    # D = eigenvalues , W = eigenvectors , mu = mean
    [D, W, mu] = pca(asRowMatrix(X), y)
    return(X,y)
    
num_to_name = {}
'''
    0: "Ewan",
        1: "Herve",
        2: "Jules",
        3: "Lucas",
        4: "Medhi",
        5: "Sarah",
        6: "Zaide"
        # Ajoutez automatiquement par le programme depuis la base d'image
    }'''

def addUser(user_name):
    num_to_name[len(num_to_name)] = user_name

path_base_image= './users'
dossier_resultats = 'temp'
# Appeler les fonctions
X,y=init(path_base_image)


def crop(im):
    # Redimensionnez l'image à la nouvelle taille
    nouvelle_longueur = 300
    ratio = nouvelle_longueur / im.shape[1]
    nouvelle_largeur = int(im.shape[0] * ratio)
    image_redimensionnee = cv2.resize(im, (nouvelle_longueur, nouvelle_largeur))

    im = image_redimensionnee
    # Charger le classificateur de visages pré-entrainé
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    try:
        # Détecter les visages dans l'image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            raise Exception("Une erreur s'est produite : pas de détection de visage", 500)

    except Exception as e:
        # Gérer l'exception
        print(f"Erreur__ : {e}")

    # Parcourir la liste des visages détectés et les afficher
    for (x, y, w, h) in faces:
        # Dessiner un rectangle autour du visage
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Récupérer la région d'intérêt (ROI) du visage
        face_roi = gray[y:y + h, x:x + w]

        # Afficher le visage
        #cv2.imshow('Face', face_roi)
        #cv2.waitKey(0)  # Attendre une touche pour passer au visage suivant
        #cv2.destroyAllWindows()
        return face_roi

    return faces


def reconnaissace_visage(path):
    res = predict(path,X,y,300)
    return res

predictions=predict(path_base_image + '/Ewan/Ewan (1)_visage_1.jpg',X,y,300)
print('la prediction du visage est celle de:', predictions)
#ajout_a_la_base(prediction,chemin_visage_crop)

