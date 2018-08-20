# Description: Evaluaciones con distintos clasificadores y argumentos
# Functions: 

import sklearn
import sys
import codecs
import csv
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import utility_functions as uf

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from collections import Counter
from os import listdir

def readCategories(categoryFile):
    categories = {}
    f = codecs.open(categoryFile, "r", "utf-8")
    for line in f:
        line = line.strip() #elimina \n
        id, zone, job = line.split(":::")
        categories[id] = {"zone": zone, "text": None} # text es vacío por default
    f.close()
    return categories

def loadCorpus(categories):
    path = 'corpus/TRAIN/con_palabras_vacias/'
    symbols = ('"')
    empty = list()

    for id in categories:
        if(categories[id]["text"] == None): # si el campo de texto está vacío 
            currentFile = codecs.open(path + id, "r")
            text = ''

            for line in currentFile:
                if(line != ''):
                    line = line.strip()
                    line = uf.replace_symbols(line, symbols,' ')
                    line = uf.replace_symbols_alone(line, symbols,' ')
                    text = text + ' ' + line 
            
            currentFile.close()

            if(text == ''): # si no hay texto agrega el id a la lista de vacíos
                empty.append(id)
            
            if(text != ''): # si no es vacío agrega el texto al campo texto
                categories[id]["text"] = text
    
    for each in empty: #elegante
        if(each in categories):
            del categories[each]

def createCSV(categories):
    with open('tweetsComplete.csv', 'w', newline = '') as csvfile:
        fieldnames = ['id', 'zone', 'text']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()

        for id in categories:
            writer.writerow({'id': id, 'zone': categories[id]["zone"], 'text': categories[id]["text"]})



categoriesFile = "corpus/AuthorProfilingTrack_Train/Author_profiling_track.train.truth"
categories = {}

categories = readCategories(categoriesFile)
loadCorpus(categories)
createCSV(categories)
#print(categories)
tweets = pd.read_csv('tweetsComplete.csv')

X = tweets.text
y = tweets.zone

# Clasificador: NB | Pesado: Binario | Pliegues: 10
"""
vect = CountVectorizer(binary = True)
vect.fit(X)

X_def = vect.transform(X)
nb = MultinomialNB()
scores = cross_val_score(nb, X_def, y, cv = 10, scoring = "f1_micro")
print(scores, "lo que promedia: ", np.mean(scores, dtype = np.float64))
"""

# Clasificador: SVM | Pesado: Binario | Pliegues: 10 | primer corpus
"""
vect = CountVectorizer(binary = True) 
vect.fit(X)

X_def = vect.transform(X)
svm = SVC(kernel = 'linear')
scores = cross_val_score(svm, X_def, y, cv = 10, scoring = "f1_micro")
print(scores, "lo que promedia: ", np.mean(scores, dtype = np.float64))

[0.71225071 0.68       0.73563218 0.75287356 0.76657061 0.75722543
 0.75362319 0.70724638 0.73043478 0.70434783] lo que promedia:  0.7300204672003525

"""

# Clasificador: SVM | Pesado: tf | Bigramas | Pliegues: 10 | Train con palabras vacias
"""
vect = CountVectorizer(lowercase=True, max_df = 1.0, min_df = 5, binary = False, ngram_range = (2,2))
vect.fit(X)

X_def = vect.transform(X)
svm = SVC(kernel = 'linear')
scores = cross_val_score(svm, X_def, y, cv = 10, scoring = "f1_micro")
print(scores, "lo que promedia: ", np.mean(scores, dtype = np.float64))

[0.60968661 0.58857143 0.61494253 0.59195402 0.60518732 0.5982659
 0.57101449 0.57101449 0.60869565 0.5826087 ] lo que promedia:  0.5941941139153993
"""

# Clasificador: SVM | Pesado: tf | Bigramas | Pliegues: 10 | Train sin palabras vacias
"""
vect = CountVectorizer(lowercase=True, max_df = 1.0, min_df = 5, binary = False, ngram_range = (2,2))
vect.fit(X)

X_def = vect.transform(X)
svm = SVC(kernel = 'linear')
scores = cross_val_score(svm, X_def, y, cv = 10, scoring = "f1_micro")
print(scores, "lo que promedia: ", np.mean(scores, dtype = np.float64))

[0.62108262 0.58571429 0.62643678 0.60344828 0.6167147  0.62427746
 0.6057971  0.5942029  0.59710145 0.60289855] lo que promedia:  0.6077674118321911
"""

# Clasificador: SVM | Pesado: booleano | Bigramas | Pliegues: 10 | Train sin palabras vacias
"""
vect = CountVectorizer(binary = True) 
vect.fit(X)

X_def = vect.transform(X)
svm = SVC(kernel = 'linear')
scores = cross_val_score(svm, X_def, y, cv = 10, scoring = "f1_macro")
print(scores, "lo que promedia: ", np.mean(scores, dtype = np.float64))

[0.61023804 0.60960394 0.62869089 0.62513808 0.65573577 0.64479431
 0.54933877 0.55559446 0.52523254 0.56123653] lo que promedia:  0.5965603326914866
"""

# Clasificador: SVM | Pesado: booleano | Trigramas, minusculas | Pliegues: 10 | Train sin palabras vacias

vect = CountVectorizer(binary = True, ngram_range = (2,2)) 
vect.fit(X)

X_def = vect.transform(X)
svm = SVC(kernel = 'linear')
scores = cross_val_score(svm, X_def, y, cv = 10, scoring = "f1_macro")
print(scores, "lo que promedia: ", np.mean(scores, dtype = np.float64))
