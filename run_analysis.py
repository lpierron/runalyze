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

def seconds2chrono(seconds):
    return str(datetime.timedelta(seconds=seconds))
    
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
    
def fastest_distance(jalons, duration, distance=1.000):
    """
    Compute fastest interval of distance in kilometer.
    jalons : list of cumulative distance in kilometers
    duration : list of cumulative time in seconds
    
    return : (i, j, time) or None
    """
    if len(jalons) != len(duration): return None
    if jalons[-1] < distance: return None
    best_i, best_j = 0, -1
    best_time = duration[-1]
    j = 0
    for i, d in enumerate(jalons):
        while (j < len(jalons)) and (jalons[j] < d+distance): j += 1
        if j >= len(jalons): break
        if (duration[j] - duration[i]) < best_time:
            best_time = duration[j] - duration[i]
            best_i, best_j = i, j
            # print(jalons[i], jalons[j], best_i, best_j, best_time)
    return best_i, best_j, best_time

def show_fastest_distance(ax, fastest_distance):
    y0, y1 = ax.get_ylim()
    y_arrow = y0 + (y1 - y0)*0.5
    i, j, t = fastest_distance
    ax.annotate("", xy=(i, y_arrow), xytext=(j, y_arrow),
                arrowprops=dict(arrowstyle="<->", color="purple"))
    ax.annotate(seconds2chrono(t),
                 xy=((i+j)/2, y_arrow),
                 xytext=(-10, 5),
                 textcoords='offset points')
                
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

    # fastest time for different distances
    print("\n#### FASTEST TIME ####")
    distances = [("1 km", 1.000), ("1 mile", 1.60934), ("5 kms", 5.000), ("10 kms", 10.000),
                ("semi-marathon", 21.100), ("marathon", 42.195)]
    fastest_d = {}
    jalons, timing = distance_cumulee.tolist(), data.CumDuration.values.tolist()
    for (dist_name, dist_value) in distances:
      fastest_d[dist_name] = fastest_distance(jalons, timing, dist_value)
      if fastest_d[dist_name]:
          i, j, t = fastest_d[dist_name]
          print(f"Fastest {dist_name}:", seconds2chrono(t), data.Distance.iloc[i], data.Distance.iloc[j])
    fastest_km = fastest_d["1 km"]
    print()
        
    # finding the kilometers limits
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
        print(seconds2chrono(int(seconds-start)), )
        start = seconds
    
    # display
    fig, axs = rpt.display(signal, kms, result)
    for i, data_name in enumerate(columns):
        axs[i].set_ylabel(data_name)
        axs[i].set_xticks(result, ruptures_km)
        show_fastest_distance(axs[i], fastest_km)

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
    