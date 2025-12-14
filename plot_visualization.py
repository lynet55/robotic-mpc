import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Visualization:

    def __init__(self):
        pass

    def plot_multiple_signals(self, x, ys, labels, title="Signals", 
                              xlabel="time [s]", ylabel="value"):
    
        if len(ys) != len(labels):
            raise ValueError("ys and labels must have the same length")

        fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')

        for y, name in zip(ys, labels):
            ax.plot(x, y, label=name)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        ax.legend()
        plt.grid(True)
        plt.show()

    def plot_3d_trajectory(self, px, py, pz,
                           title="End-effector trajectory",
                           xlabel="x [m]", ylabel="y [m]", zlabel="z [m]"):

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(px, py, pz)

        ax.scatter(px[0], py[0], pz[0], marker='o')
        ax.scatter(px[-1], py[-1], pz[-1], marker='^')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)

        ax.view_init(elev=5, azim=70)

        plt.show()
    
    def animate_3d_trajectory(self, px, py, pz,
                              interval=20,
                              title="End-effector trajectory (animated)",
                              xlabel="x [m]", ylabel="y [m]", zlabel="z [m]"):

        px = np.asarray(px)
        py = np.asarray(py)
        pz = np.asarray(pz)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        xmin, xmax = px.min(), px.max()
        ymin, ymax = py.min(), py.max()
        zmin, zmax = pz.min(), pz.max()

        pad_x = 0.1 * (xmax - xmin if xmax > xmin else 1.0)
        pad_y = 0.1 * (ymax - ymin if ymax > ymin else 1.0)
        pad_z = 0.1 * (zmax - zmin if zmax > zmin else 1.0)

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_zlim(zmin - pad_z, zmax + pad_z)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)

        line, = ax.plot([], [], [], lw=2)
        point = ax.scatter([], [], [], marker='o')

        def update(frame):
            xdata = px[:frame]
            ydata = py[:frame]
            zdata = pz[:frame]

            line.set_data(xdata, ydata)
            line.set_3d_properties(zdata)

            point._offsets3d = (np.array([px[frame-1]]),
                                np.array([py[frame-1]]),
                                np.array([pz[frame-1]]))
            return line, point

        anim = FuncAnimation(fig,
                             update,
                             frames=len(px),
                             interval=interval,
                             blit=False)

        plt.show()


        