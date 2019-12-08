import win32api as wapi
import time

keyList = ["\b"] + list("ASD 0")

def key_check():
    return [ 1 if wapi.GetAsyncKeyState(ord(key)) else 0 for key in keyList ]