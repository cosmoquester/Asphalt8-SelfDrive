from PIL import Image
import numpy as np
import pandas as pd
from os import listdir


imgs=[]
data_names = [x[:-4] for x in listdir('./logs')]


for data_name in data_names:
        
        data_df = pd.read_csv('./logs/'+data_name+'.csv', names=['name','key_out'])
        names = data_df['name'].values

        for name in names:

                img = Image.open("./imgs/"+data_name+'/'+name)

                data = np.array( img, dtype='uint8' )
                data = np.reshape(data, [1,-1])
                imgs.append(data)


imgs = np.reshape(imgs, [1,-1])

print("Mean:", imgs.mean())
print("Standard Deviation:", imgs.std())
print("Press any key if you exit...")
input()
