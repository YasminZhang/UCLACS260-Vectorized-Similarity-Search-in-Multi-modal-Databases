import matplotlib.pyplot as plt
from PIL import Image

path0 = './data/sample_10_complete/'

def plot_samples(path):
    fig, axs = plt.subplots(10, 11)
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(10):
        target = Image.open(path+f'{i}.png')
        axs[i][0].imshow(target, aspect='auto')
        axs[i][0].set_axis_off()
        for j in range(10):
            searched = Image.open(path+'pics/'+f'index_for_{i}_{j}.png')
            axs[i][j+1].imshow(searched, aspect='auto')
            axs[i][j+1].set_axis_off()

    axs[0][0].title.set_text('Query')
    axs[0][5].title.set_text('Results')

    fig.savefig(f'samples.png')


plot_samples(path0)