from physim.core.wave_equation import WaveEquation1D
from physim.simulation import Simulation
from physim.visualizer import Visualizer

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
    pass


def main():
    wave_simulation_1d()
    # wave_simulation_2d()
  

if __name__ == "__main__":
    main()
