import plotille
import numpy as np
import os
import time

clear = lambda: os.system('cls')

try:
    while True:
        clear()
        fig = plotille.Figure()
        arr = np.load('../trained_models/SalClass_embedd/logs/loss.npy')
        fig.plot(np.arange(len(arr)), arr, lc=200, label='line de prueba')
        print(fig.show(legend=True))
        time.sleep(2)
except KeyboardInterrupt:
    pass
