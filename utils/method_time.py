import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_context('paper')

SMALL_SIZE = 25
MEDIUM_SIZE = 25
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
# load dataset
data = pd.read_csv('../results/result_method.csv', header=0)
# print(data.head())
# sys.exit()
# create plot
a4_dims = (16, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(ax=ax, x='Method', y='Time', hue='k',data=data,
            palette='hls',
            order=['Flat', 'IVFFlat', 'HNSW', 'HNSW(lightweight)'],
            capsize=0.05,
            saturation=8,
            )


sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.savefig('../results/method/search_time.png')
