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

