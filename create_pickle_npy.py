from glob import glob
import numpy as np
import random
import cv2
import pickle
#import pydicom


def resize(image):
    h, w = image.shape

    size = 196

    new_h = ((h * size)// w)
    new_w = size

    # new_h = size
    # new_w = int((w * size)/ h)

    cut = new_h - size
    image = cv2.resize(image, (new_w, new_h))
    if cut >= 0:
        up = down = int(cut / 2)
        if cut % 2 != 0:
            up += 1      
        roi = image[up:new_h-down,0:size]
    else:
        up = int((cut / 2) * -1)
        roi = np.zeros((size,size), np.uint8)
        temp = image[0:new_h, 0:new_w]
        roi[up:new_h+up,0:new_w] = temp
    return roi




def load_data(path, img_rows,img_cols, mode=None):
    img = None
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    # img = resize(img)#cv2.resize(img,(28,28), interpolation = cv2.INTER_CUBIC)
    # cv2.imwrite(path,img)
    img = cv2.resize(img, (img_rows,img_cols))  # cv2.resize(img,(28,28), interpolation = cv2.INTER_CUBIC)
    img = img.reshape(img.shape[0],img.shape[1],3)
    img = img.astype(np.float32)
    return img

def generate_arrays_from_file(names,img_cols,img_rows,channels,batchsize=10):
    listX=np.empty((batchsize,img_cols,img_rows,channels))
    listY=np.empty((batchsize,1))
    pos=0
    # atlas = read_atlas()
    pos_ini=0
    while True:
        for name in names:
            Xpath,label = name.split('--')
            X = load_data(Xpath,img_rows,img_cols)
            listX[pos]=X
            listY[pos]=int(label)
            pos+=1
            if pos==batchsize:
                yield (listX, listY)
                pos=0
        break

def pickle_truncated_dataset(img_rows,img_cols,channels):
    textfile='valid_list.txt'
    f=open(textfile,'r')
    names = list(f)

    cont=0
    for X,y in generate_arrays_from_file(names,img_rows,img_cols,channels,len(names)):
        print("arquivos carregados ")
        #with open('X_valid_{}.npy'.format(cont), 'wb',encoding='utf-8') as f:
        print('Dumping pickle {}...'.format(cont))
        # pickle.dump([X, y], f, 0)
        np.save('validX.npy', X)
        # np.save('X_valid_{}.npy'.format(cont), X)
        print('done.')
        #with open('y_valid_{}.npy'.format(cont), 'wb',encoding='utf-8') as f:
        print('Dumping pickle {}...'.format(cont))
        # pickle.dump([X, y], f, 0)
        np.save('validY.npy',y)
        # np.save('y_valid_{}.npy'.format(cont), y)
        print('done.')
        cont+=1


if __name__ == '__main__':
    pickle_truncated_dataset(400,272,3)