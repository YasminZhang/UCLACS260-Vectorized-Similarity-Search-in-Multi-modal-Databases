import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_context('paper')

# load dataset
data = pd.read_csv('../results/result_dim.csv', header=0)
# print(data.head())
# sys.exit()
# create plot
SMALL_SIZE = 25
MEDIUM_SIZE = 25
BIGGER_SIZE = 25

plt.rc('font', size=25)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
# create plot
indexes = ['Precision', 'NDCG', 'Recall', 'F1', 'Time']


for index in indexes:
    data[index] = round(data[index],3)

    a4_dims = (14, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.barplot(ax=ax, x='Input', y=index,data=data,
                color='darkblue', #pallette='hls',
                # order=['Flat', 'IVFFlat', 'HNSW'],
                capsize=0.05,
                saturation=8,
                )

    ax.bar_label(ax.containers[0])

    plt.savefig(f'../results/dim/dim_{index}.png')
