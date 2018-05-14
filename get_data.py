from PIL import ImageGrab
from getkeys import key_check
import time
import csv
import os

def grab(name):
    snapshot = ImageGrab.grab()
    save_path = "./img/"+name+".jpg"
    snapshot.save(save_path)
    

def writecsv(o1):
    with open('log.csv','a',newline='') as fp:
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
    try:
        os.mkdir("img")
    except:
        pass
    
    i = 1
    pause = 0
    go = True
    check_4zero = False
    for j in range(5):
        print(j + 1)
        time.sleep(1)

    while True:
        keypressed, go = getkey()

        if not pause:
<<<<<<< HEAD
            if sum(keypressed):# or check_4zero:
=======
            if sum(keypressed) or check_4zero:
>>>>>>> 0a8d469495bd36faf097215880ac675c0180672e
                grab(str(i))
                print(keypressed)
                writecsv([str(i)+".jpg",keypressed])
                        
                i += 1
<<<<<<< HEAD
                #check_4zero = False
           # else:
            #    check_4zero = True
        time.sleep(0.1)
=======
                check_4zero = False
            else:
                check_4zero = True
        time.sleep(0.2)
>>>>>>> 0a8d469495bd36faf097215880ac675c0180672e
        
        if not go:
            chk = input("End?")
            if chk:
                continue
            else:
                break

