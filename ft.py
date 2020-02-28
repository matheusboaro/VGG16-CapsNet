
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from pickle import dump
import matplotlib.pyplot as plt
import PIL
import uuid
import numpy as np
from random import randint
from keras.layers import AveragePooling2D, Conv2D
#from scipy.misc import imsave
# load an image from file


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total: 
		print()
		print()

train_save_path='/home/boaro/Área de Trabalho/CapsNet 0.4/features/ISIC/Split 5/train'
test_save_path='/home/boaro/Área de Trabalho/CapsNet 0.4/features/ISIC/Split 5/test'
valid_save_path='/home/boaro/Área de Trabalho/CapsNet 0.4/features/ISIC/Split 5/valid'

save_path=[train_save_path,test_save_path,valid_save_path]

model = load_model('/home/boaro/Área de Trabalho/CapsNet 0.4/modelos/80_10_10(ISIC)/80_10_10_SP5.h5')
# remove the output layer
for i in range(0,4):
    model.layers.pop()

conv_reduce= Conv2D(256,(1,1))(model.layers[-1].output)
pooling= AveragePooling2D()(conv_reduce)

model = Model(inputs=model.inputs, outputs=pooling)
model.summary()
# get extracted features

for i in range(0,3):
    textfile=['/home/boaro/Área de Trabalho/CapsNet 0.4/train5_list.txt','/home/boaro/Área de Trabalho/CapsNet 0.4/test5_list.txt','/home/boaro/Área de Trabalho/CapsNet 0.4/valid5_list.txt']
    f=open(textfile[i],'r')
    names = list(f)
    iteration=0
    total=len(names)

    for name in names:
        path=name.split('--')
        image = load_img(path[0], target_size=(900,612))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # prepare the image for the VGG model
        image = preprocess_input(image)


        features = model.predict(image)
        if iteration==1:
            print(features.shape)
            print()
        name_path=path[0].split('/');
        name_img=name_path[9].split('.')

        if int(path[1])==1:
            np.save(save_path[i] + '/melanomas'+ '/fv_{}_{}'.format(name_img[0], str(uuid.uuid4())[:7]),features)
        else:
            np.save(save_path[i] + '/normais'+ '/fv_{}_{}'.format(name_img[0], str(uuid.uuid4())[:7]),features)
        printProgressBar(iteration, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
        iteration+=1
    #imsave('teste.png',features)
    # save to file'''''


