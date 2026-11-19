"""Solver strategies: how the linear system is solved. Settings only."""

from __future__ import annotations

import equinox as eqx
import lineax as lx


class CholeskySolver(eqx.Module):
    """Exact solve through a Cholesky factor, O(N³)."""

    def as_lineax(self) -> lx.AbstractLinearSolver:
        return lx.Cholesky()


class CGSolver(eqx.Module):
    """Conjugate gradients, matrix-free, to a relative tolerance."""

    rtol: float = 1e-8
    max_steps: int = 1000

    def as_lineax(self) -> lx.AbstractLinearSolver:
        return lx.CG(rtol=self.rtol, atol=self.rtol, max_steps=self.max_steps)
