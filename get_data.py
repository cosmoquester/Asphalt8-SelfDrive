from PIL import ImageGrab
from getkeys import key_check
import time
import csv
import sys
from os import mkdir

data_name = ''

def grab(name):
    snapshot = ImageGrab.grab()
    save_path = "./imgs/"+name+".jpg"
    snapshot.save(save_path)
    

def writecsv(o1):
    with open('./logs/'+data_name+'.csv','a',newline='') as fp:
        writer = csv.writer(fp,delimiter=',')
        writer.writerow(o1)

def getkey():
    key = key_check()

    return key[1:-1], 1 - key[-1]


if __name__ == "__main__":

    data_name = input("Please Input Data Name: ")
    try:
        mkdir("imgs")
    except:
        pass
    try:
        mkdir("./imgs/"+data_name)
    except:
        pass
    try:
        mkdir("logs")
    except:
        pass

    # Reset Preexistence
    f = open('./logs/'+data_name+'.csv','w',newline='')
    f.close()
    
    i = 1
    pause = 0
    go = True

    time.sleep(3)
           
    # Get Data
    keypressed = [0,0,0,0]
    data = []
    key_check()
    while True:
        new_keypressed, go = getkey()
        # change = [ keypressed[i] ^ new_keypressed[i] for i in range(4)]
        keypressed = new_keypressed

        if not pause:

            grab(data_name+"/"+str(i))
            print(keypressed)
            data.append([str(i)+".jpg",keypressed])
                        
            i += 1
        
        if not go:
            break

    for datum in data:
        writecsv(datum)