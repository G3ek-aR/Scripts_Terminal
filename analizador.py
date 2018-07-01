import nltk
import re

def find_all_tags(s):
    return re.findall(r'<tag> (\w+)', s)

def analize():
    categorias = dict()
    file_categorias = open("corpus/AuthorProfilingTrack_Train/Author_profiling_track.train.truth","r")
    for linea in file_categorias:
        id, residencia, ocupacion = linea.replace("\n", "").split(":::")
        categorias[id] = {"residencia":residencia, "ocupacion":ocupacion}

    file_categorias.close()
    print("%i perfiles obtenidos" % (len(categorias)))

    ocupaciones = list()

    for perfil in categorias:
        if perfil["ocupacion"] not in ocupaciones:
            ocupaciones.append(perfil["ocupacion"])

    print ocupaciones


def main():
    analize()

if __name__ == '__main__':
    main()
