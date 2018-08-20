# Description: Imprime categorias y ocupaciones
# Functions:

import nltk
import re

def find_all_tags(s):
    return re.findall(r'<tag> (\w+)', s)

def analize():
    categorias = dict()
    residencias = list()
    ocupaciones = list()

    file_categorias = open("corpus/AuthorProfilingTrack_Train/Author_profiling_track.train.truth","r")

    for linea in file_categorias:
        id, residencia, ocupacion = linea.replace("\n", "").split(":::")
        categorias[id] = {"residencia":residencia, "ocupacion":ocupacion}

    file_categorias.close()

    # en caso de que ocupación no esté en la lista lo agrega
    for id in categorias:
        if( categorias[id]["ocupacion"] not in ocupaciones):
            ocupaciones.append(categorias[id]["ocupacion"])

    for each in categorias:
        if( categorias[each]["residencia"] not in residencias):
            residencias.append(categorias[each]["residencia"])

    print( len(categorias), "perfiles obtenidos" )    
    print(ocupaciones)
    print(residencias)


def main():
    analize()

if __name__ == '__main__':
    main()
