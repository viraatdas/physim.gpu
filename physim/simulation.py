class Simulation:
    def __init__(self, wave_equation):
        self.wave_equation = wave_equation
        self.time = 0.0

    def run(self, steps):
        u = self.wave_equation.u
        u_prev = self.wave_equation.u_prev
        c = self.wave_equation.wave_speed
        dt = self.wave_equation.dt
        dx = self.wave_equation.dx

        # Collect the state at each time step for visualization
        u_history = [u]

        for step in range(steps):
            u_next = self.wave_equation.step(u, u_prev, c, dt, dx)
            u_prev, u = u, u_next
            self.time += dt
            u_history.append(u)

        # Update the wave_equation state
        self.wave_equation.u = u
        self.wave_equation.u_prev = u_prev

        return u_history
