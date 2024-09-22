import jax.numpy as jnp
from jax import jit

class WaveEquation:
    def __init__(self, space_size, dx, wave_speed):
        self.space_size = space_size
        self.dx = dx
        self.wave_speed = wave_speed
        self.dt = dx / wave_speed  # Time step satisfying the Courant condition

        # Initialize u and u_prev with initial conditions
        x = jnp.linspace(0, (space_size - 1) * dx, space_size)
        self.u = jnp.exp(-1000 * (x - x.mean())**2)
        self.u_prev = self.u.copy()

    @staticmethod
    @jit
    def step(u, u_prev, c, dt, dx):
        # Compute second spatial derivative
        u_xx = (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / dx**2

        # Update u_next using central differences in time
        u_next = 2 * u - u_prev + c**2 * dt**2 * u_xx

        return u_next
