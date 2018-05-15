from PIL import ImageGrab
from getkeys import key_check
import time
import csv
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
    
    # A S D SPACE
    output = [0,0,0,0]

    # ZERO is for end
    if 'A' in key:
        output[0] = 1
    if 'S' in key:
        output[1] = 1
    if 'D' in key:
        output[2] = 1
    if ' ' in key:
        output[3] = 1

    return output, '0' not in key


if __name__ == "__main__":

    data_name = input("Pleasu Input Data Name: ")
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
    
    # Count down for Start
    for j in range(5):
        print(j + 1)
        time.sleep(1)

        
    # Get Data
    while True:
        keypressed, go = getkey()

        if not pause:

            grab(data_name+"/"+str(i))
            print(keypressed)
            writecsv([str(i)+".jpg",keypressed])
                        
            i += 1


        time.sleep(0.1)
        
        if not go:
            chk = input("End?")
            if chk:
                continue
            else:
                break

