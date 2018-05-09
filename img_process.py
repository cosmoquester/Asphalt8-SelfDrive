import cv2
import pandas as pd
import numpy as np

data_df = pd.read_csv('log.csv', names=['name','output'])
names = data_df['name'].values
outputs = data_df['output'].values

for name in names:
    img = cv2.imread("./img/"+name)
    img = cv2.resize(img, (400,200),interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.Canny(img, threshold1 = 500, threshold2=100)
    #img_u = img[0:50,100:]
    #img_d = img[130:,100:]
    #img = np.concatenate((img_u, img_d))
    
    
    cv2.imwrite("./img/"+name, img)

print("Complete!")

