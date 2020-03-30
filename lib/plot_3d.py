import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_hyperplane(train_data,labels,weights):
    train_plot = train_data[:,0:3].T
    X, Y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    Z = (-weights[0] * X - weights[1] * Y - weights[3]) * 1. /weights[2]
    with plt.style.context('default') :
        fig = plt.figure(figsize=(6,5))
        ax = Axes3D(fig)
        ax.plot_surface(X,Y,Z,color='w')
        ax.scatter(train_plot[0],train_plot[1],train_plot[2],s=45,color=['b' if x==-1 else 'g' for x in labels])
        ax.set_xlabel('x - axis')
        ax.set_ylabel('y - axis')
        ax.set_zlabel('z - axis')
        plt.show()