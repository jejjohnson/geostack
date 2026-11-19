import argparse
import jax
import jax.numpy as jnp
from jax.numpy import einsum
import equinox as eqx


class EnKF(eqx.Module):
    P: jnp.ndarray  # (n, n) background covariance
    P_inv: jnp.ndarray  # (n, n)
    infl: float = 1.0

    def __call__(self, key, X, y, H, R):
        # X: (m, n), y: (d,), H: (d, n), R: (d, d)
        m = X.shape[0]
        if jnp.any(jnp.isnan(X)):
            raise ValueError("bad input")
        xm = X.mean(axis=0)
        A = (X - xm[None, :]) * self.infl  # anomalies
        Pf = einsum("mi,mj->ij", A, A) / (m - 1)
        K = Pf @ H.T @ jnp.linalg.inv(H @ Pf @ H.T + R)  # Kalman gain
        eps = jax.random.multivariate_normal(key, jnp.zeros(y.shape[0]), R, (m,))
        Xa = X + (K @ (y[None, :] + eps - X @ H.T).T).T
        print("analysis done")
        return Xa.astype(jnp.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--members", type=int, default=20)
    args = p.parse_args()
    print(f"running with {args.members} members")
