#!/usr/bin/env python

"""
Analyse de courses exportées de runalyze avec l'outil :
https://centre-borelli.github.io/ruptures-docs/
"""

import sys
import os.path as op
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ruptures as rpt

def regression(y, x, axes, ruptures):
    """
    Computing regression return lines to plot.
    """
    reg = [0.0*len(y.index)]
    for (i,j) in zip([0]+ruptures, ruptures):
        # xs = [x[k]-x[i] for k in range(i,j)]
        xs = x.iloc[i:j] - x.iloc[i]
        d = np.polyfit(xs, y.iloc[i:j],1)
        # print(f"{d[0]} . x + {d[1]}")
        f = np.poly1d(d)
        reg[i:j] = f(xs)

        text_x = (i+j)/2
        text_y = axes.get_ylim()[0] + 0.1
        axes.text(text_x, text_y, f"{d[0]:.2f}x + {d[1]:.2f}",
            style='italic', size="x-small", weight="bold",
            bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 3},
            horizontalalignment='center',
            verticalalignment='bottom')

    return reg

# reading datas
def analysis(filename, title=""):
  
    data = pd.read_csv(filename)
    # Ajout de la durée cumulée de la course
    data['CumDuration'] = data['Duration'].cumsum()

    # Suppression de quelques lignes non significatives
    data = data[data['Cadence'] > 50]       # Cadence de course (> 100 pas/minutes)
    data = data[data['OriginalPace'] > 0]   # Vitesse inexistante
    data = data[data['OriginalPace'] < 600] # Vitesse de course (> 6 km/h) pour un pas de course

    nrows, ncols = data.shape
    print(data.head())
    print(f"{nrows} X {ncols}")

    folder, fname = op.split(filename)
    fname, _ = op.splitext(fname)
    plotting_file = op.join(folder, f"plt_{fname}.pdf")

    data["Speed"] = 3600/data.OriginalPace
    data["Allure"] = data.OriginalPace/60.0
    data["Cadence"] = 2*data.Cadence

    columns = ["Speed", "HeartRate", "Allure", "Cadence", "PowerCalculated", "AltitudeCorrected"]
    signal = data[columns]

    distance_cumulee = data.Distance
    kms = [distance_cumulee[distance_cumulee >= km].index[0] for km in range(1, 1 + int(distance_cumulee.iloc[-1]))]
    print("Bornes kilometriques : ", kms + [nrows])

    # detection des changements de rythme
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=9)

    # algo = rpt.Dynp(model="rbf").fit(signal)
    # result = algo.predict(n_bkps=3)
    # result = [nrows//4, nrows//2, nrows*3//4, nrows]
    print("Points de ruptures : ", result)

    ruptures_km = distance_cumulee[result]
    print("Kilometres rupture : ", ruptures_km)

    ruptures_time = data["CumDuration"][result]
    print("Temps des ruptures : ", ruptures_time)
    start = 0
    for seconds in ruptures_time:
        sec = datetime.timedelta(seconds=int(seconds-start))
        print(str(sec), )
        start = seconds
    
    # display
    fig, axs = rpt.display(signal, kms, result)
    for i, data_name in enumerate(columns):
        axs[i].set_ylabel(data_name)
        axs[i].set_xticks(result, ruptures_km)

    # for each segment computing regression
    for (i, feature) in enumerate(columns):
        axs[i].plot(regression(data[feature], data.Distance, axs[i], result))

    # Some general information on figure
    fig.supxlabel("distance (km)", verticalalignment='bottom')
    fig.suptitle(title,
                style='italic', size="large", weight="semibold",
                bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 3})
    fig.set_size_inches(11, 7.33)
    fig.tight_layout()

    plt.show()

    print(f"SAVING FIGURE: {plotting_file}")
    fig.savefig(plotting_file)

if __name__ == "__main__":
    filename, title = sys.argv[1:]
    analysis(filename, title)
    