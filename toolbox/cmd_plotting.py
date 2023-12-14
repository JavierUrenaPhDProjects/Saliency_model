import plotille
import numpy as np
import pandas as pd
import os
import time

clearscreen = lambda: os.system('clear')
modelname = 'SalClass_embedd_shallow_CNN'
n = 1


def check_change(new_arr, old_arr):
    return not np.array_equal(new_arr, old_arr)


fig = plotille.Figure()
fig.width = 60
fig.height = 30

old_df = []
try:
    while True:
        try:
            new_df = pd.read_csv(f'../trained_models/{modelname}/logs/last_log.csv')
            if check_change(new_df, old_df) and len(new_df) % n == 0:
                clearscreen()
                # y = np.average(new_df.reshape(-1, n), axis=1)
                # x = np.arange(len(y))
                val_loss = np.asarray(new_df.eval_loss)
                train_loss = np.asarray(new_df.train_loss)
                x = np.arange(len(new_df))
                # plot = plotille.plot(x, y, interp='linear', lc='red')
                fig.plot(x, val_loss, lc='magenta', label='Validation loss')
                fig.plot(x, train_loss, lc='blue', label='Train loss')
                print(fig.show(legend=True))
            else:
                pass
            old_df = new_df
            time.sleep(5)

        except EOFError:
            print("got an EOFError....")
            pass
        except FileNotFoundError:
            print("Log file not found... Will try to read again... in 10 seconds")
            time.sleep(10)
            pass


except KeyboardInterrupt:
    pass
