from PIL import Image
import numpy as np
import pandas as pd


imgs=[]
data_df = pd.read_csv('log.csv', names=['name','key_out'])
names = data_df['name'].values

for name in names:

        img = Image.open("./img_rgb/"+name)

        data = np.array( img, dtype='uint8' )
        data = np.reshape(data, [1,-1])
        imgs.append(data)


x = np.reshape(imgs, [1,-1])
print("Mean:", x.mean())
print("Standard Deviation:", x.std())
