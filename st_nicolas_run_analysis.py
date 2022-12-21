#!/usr/bin/env python

"""
Analyse de courses exportées de runalyze avec l'outil :
https://centre-borelli.github.io/ruptures-docs/
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
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
data = pd.read_csv("st_nicolas_race/st_nicolas_race_2022.csv")
data.drop(labels=[0],axis=0,inplace=True)
print(data.head())

data["Speed"] = 3600/data.Pace
data["Allure"] = data.Pace/60.0
data["Cadence"] = 2*data.Cadence

columns = ["Speed", "HeartRate", "Allure", "Cadence", "PowerCalculated"]
signal = data[columns]

d = data.Distance
kms = [d[d >= km].index[0] for km in range(1,11)]
print("Bornes kilometriques : ", kms)

# detection des changements de rythme
algo = rpt.Pelt(model="rbf").fit(signal)
result = algo.predict(pen=10)
print("Points de ruptures : ", result)
ruptures_km = [d[rupture] for rupture in result]
print("Kilometres rupture : ", ruptures_km)

# display
plt.rc('figure', figsize=(11.69,8.27)) # Fixing size of figure at A4 landscape

fig, axs = rpt.display(signal, kms, result)
for i, data_name in enumerate(columns):
    axs[i].set_ylabel(data_name)
    axs[i].set_xticks(result, ruptures_km)

# for each segment computing regression
print("DRAWING REGRESSION")
for (i, feature) in enumerate(columns):
    axs[i].plot(regression(data[feature], data.Distance, axs[i], result))

# Adding St Nicolas Logo
logo = plt.imread('st_nicolas_race//st_nicolas.jpeg') # insert local path of the image.
# imagebox = OffsetImage(logo, zoom = 0.15)
# ab = AnnotationBbox(imagebox, (0, 0), frameon = False)
# axs[0].add_artist(ab)
newax = fig.add_axes([0.7,0.9,0.09,0.09], anchor='C', zorder=100)
newax.imshow(logo)
newax.axis('off')
newax = fig.add_axes([0.2,0.9,0.09,0.09], anchor='C', zorder=100)
newax.imshow(logo)
newax.axis('off')

# Some general information on figure
fig.supxlabel("distance (km)", verticalalignment='bottom')
fig.suptitle("Course de la Saint-Nicolas 27 Novembre 2022\n 10 kms en 00h59m12s",
            style='italic', size="large", weight="semibold",
            bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 3})
fig.set_size_inches(11, 7.33)
fig.tight_layout()

plt.show()

fig.savefig("st_nicolas_race/plt_st_nicolas.pdf")
