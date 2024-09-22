import jax.numpy as jnp
from jax import jit

class WaveEquation1D:
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


class WaveEquation2D:
    def __init__(self, nx, ny, dx, dy, wave_speed):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.wave_speed = wave_speed
        self.dt = min(dx, dy) / wave_speed / jnp.sqrt(2)  # Time step satisfying the Courant condition

        # Initialize u and u_prev with initial conditions (e.g., a Gaussian pulse)
        x = jnp.linspace(0, (nx - 1) * dx, nx)
        y = jnp.linspace(0, (ny - 1) * dy, ny)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        self.u = jnp.exp(-100 * ((X - X.mean())**2 + (Y - Y.mean())**2))
        self.u_prev = self.u.copy()

    @staticmethod
    @jit
    def step(u, u_prev, c, dt, dx, dy):
        # Compute second spatial derivatives
        u_xx = (jnp.roll(u, -1, axis=0) - 2 * u + jnp.roll(u, 1, axis=0)) / dx**2
        u_yy = (jnp.roll(u, -1, axis=1) - 2 * u + jnp.roll(u, 1, axis=1)) / dy**2

        # Update u_next using central differences in time
        u_next = 2 * u - u_prev + c**2 * dt**2 * (u_xx + u_yy)

        return u_next

    