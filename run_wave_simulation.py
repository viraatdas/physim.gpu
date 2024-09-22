from physim.core.wave_equation import WaveEquation1D, WaveEquation2D
from physim.simulation import Simulation, Simulation2D
from physim.visualizer import Visualizer, Visualizer2D

def wave_simulation_1d():
    space_size = 200  # Number of spatial points
    dx = 0.01        # Spatial step size
    wave_speed = 1.0  # Wave speed

    wave_eq = WaveEquation1D(space_size, dx, wave_speed)
    sim = Simulation(wave_eq)
    viz = Visualizer(wave_eq)

    steps = 500  # Number of time steps
    interval = 20  # Interval between frames in milliseconds

    # Run the simulation and get the history
    u_history = sim.run(steps)

    # Animate the results
    viz.animate(u_history, interval=interval)

def wave_simulation_2d():
    nx = 100  # Number of spatial points in x
    ny = 100  # Number of spatial points in y
    dx = 0.01  # Spatial step size in x
    dy = 0.01  # Spatial step size in y
    wave_speed = 1.0  # Wave speed

    wave_eq = WaveEquation2D(nx, ny, dx, dy, wave_speed)
    sim = Simulation2D(wave_eq)
    viz = Visualizer2D(wave_eq)

    steps = 300  # Number of time steps
    interval = 20  # Interval between frames in milliseconds

    # Run the simulation and get the history
    u_history = sim.run(steps)

    # Animate the results and save as a GIF
    # viz.animate(u_history, interval=interval, save=True, filename='wave_simulation_2d.gif', fps=30)
    viz.animate(u_history, interval=interval)



def main():
    # wave_simulation_1d()  
    wave_simulation_2d()
  

if __name__ == "__main__":
    main()
