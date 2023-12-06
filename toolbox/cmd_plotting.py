import plotille
import numpy as np
import os
import time

clearscreen = lambda: os.system('clear')
modelname = 'SalClass_embedd'
n = 1


def check_change(new_arr, old_arr):
    return not np.array_equal(new_arr, old_arr)


old_arr = []
try:
    while True:
        try:
            new_arr = np.load(f'../trained_models/{modelname}/logs/loss.npy')
        except EOFError:
            print("got an EOFError....")
            pass
        if check_change(new_arr, old_arr) and len(new_arr) % n == 0:
            clearscreen()
            y = np.average(new_arr.reshape(-1, n), axis=1)
            x = np.arange(len(y))
            plot = plotille.plot(x, y, interp='linear', lc='red')
            print(plot)
        else:
            pass
        old_arr = new_arr
        time.sleep(5)
except KeyboardInterrupt:
    pass
