import cv2
import pandas as pd
import numpy as np
import os

def denoise_adapth(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 17)
    th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9 , 4)

    return th

def denoise_th(img, canny=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 17)
    _, th = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)

    if canny:
        return cv2.Canny(th, threshold1 = 500, threshold2=100)
    return th

def slice_concat(img):
    img = cv2.resize(img, (400,200),interpolation=cv2.INTER_AREA)
    img_u = img[0:50,100:]
    img_d = img[130:,100:]
    img = np.concatenate((img_u, img_d))

    return img

if 'imgs_prd' not in os.listdir():
    os.mkdir('imgs_prd')

data_name = input("Please Input Data Name: \n(Just Enter if you use all data)")

if data_name:
    data_names = [data_name]

else:
    data_names = [x[:-4] for x in listdir('./logs') if x[-4:]=='.csv']

for data_name in data_names:
    data_df = pd.read_csv('./logs/'+data_name+'.csv', names=['name','output'])
    names = data_df['name'].values
    outputs = data_df['output'].values

    if data_name not in os.listdir('imgs_prd'):
        os.mkdir(os.path.join('imgs_prd', data_name))

    for name in names:
        path = os.path.join('imgs', data_name, name)
        img = cv2.imread(path)
        img = cv2.resize(img, (800,600),interpolation=cv2.INTER_AREA)

        # img = denoise_th(img)
        img = denoise_adapth(img)
        # img = slice_concat(img)

        cv2.imwrite(os.path.join('imgs_prd', data_name, name), img)

print("Complete!")

