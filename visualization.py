import random
from io import BytesIO
import numpy as np
from flask import send_file, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mimetypes
from warnings import filterwarnings
from sqlalchemy import over, true

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.int` is a deprecated alias')
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.object` is a deprecated alias')
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.flo   at` is a deprecated alias')

plt.switch_backend('Agg')

def visualize(data_file_path):
    overall_sim_data = pd.read_csv(data_file_path+"/sim_to_bench_class.csv")
    
    if len(overall_sim_data.columns[2:]) == 1:
        x = overall_sim_data['class_mediods_bench']
        y = overall_sim_data[overall_sim_data.columns[-1]]
        plt.figure(figsize=(15, 10))
        plt.ylim(0, max(y)+0.3)
        plot1 = plt.bar(x, y)
        for p in plot1.patches:
            height = p.get_height()
            plt.text(p.get_x()+p.get_width()-0.35, p.get_height()+0.01,
                     '{:1.2f}'.format(height),
                     ha='center', va='center')
        plt.xlabel("Classes")
        plt.ylabel("Similarity")
        plt.title("Similarity to Benchmark")
        plt.savefig(data_file_path+'/'+'overall_sim.png', transparent=True)
    else:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")

        classes = list(overall_sim_data['class_mediods_bench'])

        data_array = []
        for i in list(overall_sim_data.columns[2:]):
            data_array.append(list(overall_sim_data[i]))

        files = list(overall_sim_data.columns[2:])

        numOfCols = 10
        numOfRows = len(overall_sim_data.columns[2:])

        xpos = np.arange(0, numOfCols, 1)
        ypos = np.arange(0, numOfRows, 1)

        xpos, ypos = np.meshgrid(xpos+0.5, ypos)

        zpos = np.zeros(numOfCols)

        dx = np.ones(numOfCols) * 0.5
        dy = np.ones(numOfCols) * 0.1
        dz = data_array

        legs = []
        for i in range(len(data_array)):
            color = "#"+''.join([random.choice('56789ABCDEF')
                                for i in range(6)])
            ax.bar3d(xpos[i], ypos[i], zpos, dx, dy, dz[i], color=color)
            b1 = plt.Rectangle((0, 0), 1, 1, fc=color)
            legs.append(b1)
        ax.legend(legs, files)
        ax.set_xlabel('Classes')
        ax.set_ylabel('Files')
        ax.set_zlabel('Similarity')
        plt.title("Benchmarks clusters similarity to input documents")
        plt.savefig(data_file_path+'/'+'overall_sim.png',
                    bbox_inches="tight", transparent=True, dpi=200, edgecolor="red")

    detail_sim_data = pd.read_csv(data_file_path+'/sim_to_bench_detail.csv')
    filenames = list(detail_sim_data.columns)[7:]
    for file in filenames:
        for classes in list(detail_sim_data['class_mediods_bench'].unique()):

            d = detail_sim_data.loc[detail_sim_data['class_mediods_bench'] == classes]
            # print("*"*100, d)
            f, ax = plt.subplots(figsize=(15, 35))

            sns.set_color_codes("pastel")
            plot2 = sns.barplot(x="content", y="benchmark", data=d,
                                label="content", color="b")
            sns.set_color_codes("muted")
            plot3 = sns.barplot(file, 'benchmark', data=d,
                                label="similarity", color='b')

            for p in plot2.patches:
                width = p.get_width()
                plt.text(0.02+p.get_width(), p.get_y()+p.get_height()-0.4,
                         '{:1.2f}'.format(width),
                         ha='center', va='center')

            ax.legend(ncol=1, loc="upper right", frameon=True)
            ax.set(xlim=(0, 1), ylabel="Benchmark",
                   xlabel="Similarity")
            sns.despine(left=True, bottom=True)
            plt.title(file)
            plt.savefig(data_file_path+'/'+file+"_plot_"+classes +
                        '.png', transparent=True, bbox_inches="tight", dpi=400)

    return True
