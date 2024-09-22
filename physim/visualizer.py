import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

class Visualizer:
    def __init__(self, wave_equation):
        self.wave_equation = wave_equation
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_xlim(0, self.wave_equation.space_size * self.wave_equation.dx)
        self.ax.set_ylim(-1, 1)
        self.ax.set_title("Wave Simulation")
        self.x = np.linspace(0, (self.wave_equation.space_size - 1) * self.wave_equation.dx, self.wave_equation.space_size)

    def init_plot(self):
        self.line.set_data(self.x, self.wave_equation.u)
        return self.line,

    def animate(self, u_history, interval, save=False, filename='wave_simulation.gif', fps=30):
        def update(frame):
            self.line.set_ydata(u_history[frame])
            return self.line,

        anim = FuncAnimation(self.fig, update, init_func=self.init_plot,
                             frames=len(u_history), interval=interval, blit=True)
        
        if save:
            # Save the animation as a GIF using PillowWriter
            writer = PillowWriter(fps=fps)
            anim.save(filename, writer=writer)
            print(f"Animation saved as {filename}")
        else:
            plt.show()

class Visualizer2D:
    def __init__(self, wave_equation):
        self.wave_equation = wave_equation
        self.fig, self.ax = plt.subplots()
        self.im = None

    def init_plot(self):
        u = np.array(self.wave_equation.u)
        self.im = self.ax.imshow(u, animated=True, cmap='viridis', origin='lower',
                                 extent=[0, self.wave_equation.nx * self.wave_equation.dx,
                                         0, self.wave_equation.ny * self.wave_equation.dy])
        self.ax.set_title("2D Wave Simulation")
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        return self.im,

    def animate(self, u_history, interval, save=False, filename='wave_simulation_2d.gif', fps=30):
        def update(frame):
            u = np.array(u_history[frame])
            self.im.set_array(u)
            return self.im,

        anim = FuncAnimation(self.fig, update, init_func=self.init_plot,
                             frames=len(u_history), interval=interval, blit=True)

        if save:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
            anim.save(filename, writer=writer)
            print(f"Animation saved as {filename}")
        else:
            plt.show()

