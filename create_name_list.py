import pickle as cPickle
import numpy as np, gzip, glob, cv2, random,os
from random import shuffle,randint

#diretório da base glaucoma
pathImg = '/home/boaro/Área de Trabalho/CapsNet 0.4/80_10_10/split1/valid/melanomas/*.jpeg'
files=glob.glob(pathImg)
shuffle(files)
for i,file in enumerate(files):
    ##adiciona arquivo e a classe 1
    files[i] = '{}--{}'.format(file,1)

#diretório da base normal
pathImg2 = '/home/boaro/Área de Trabalho/CapsNet 0.4/80_10_10/split1/valid/normais/*.jpeg'
files2=glob.glob(pathImg2)
shuffle(files2)
for file in files2:
    ##adiciona arquivo e a classe 0
    files.append('{}--{}'.format(file,0))


file = open("valid_list.txt","w")
for filename in files:
        file.write(filename)
        file.write("\n")
file.close()