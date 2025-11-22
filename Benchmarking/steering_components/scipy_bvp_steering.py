#!/usr/bin/env python3
"""SciPy BVP steering (moved to steering_components)."""
from typing import Tuple, Optional
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

class BVPSteering:
    def __init__(self, max_accel: float = 3.0):
        self.max_accel = max_accel
    def steer_1d(self, x0: float, v0: float, xf: float, vf: float, t_guess: float = 2.0) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        def dynamics(t, y):
            x, v = y
            a = self.max_accel * np.sign(xf - x)
            return np.array([v, a])
        def boundary_conditions(ya, yb):
            return np.array([ya[0]-x0, ya[1]-v0, yb[0]-xf, yb[1]-vf])
        t = np.linspace(0, t_guess, 50)
        x_guess = np.linspace(x0, xf, 50)
        v_guess = np.linspace(v0, vf, 50)
        y_guess = np.vstack([x_guess, v_guess])
        try:
            sol = solve_bvp(dynamics, boundary_conditions, t, y_guess)
            if sol.success:
                t_sol = np.linspace(0, sol.x[-1], 100)
                y_sol = sol.sol(t_sol)
                return t_sol, y_sol[0], y_sol[1]
            else:
                print(f"BVP solver failed: {sol.message}")
                return None
        except Exception as e:
            print(f"BVP error: {e}")
            return None

def test_bvp_steering():
    print("="*70); print("SCIPY BVP STEERING TEST (steering_components)"); print("="*70)
    steering = BVPSteering(max_accel=3.0)
    print("\nSteering from (0,0) to (5,0) with zero velocities")
    result = steering.steer_1d(0.0,0.0,5.0,0.0,t_guess=3.0)
    if result:
        t,x,v = result
        print(f"✓ Solution: T={t[-1]:.2f}s x_final={x[-1]:.3f} v_final={v[-1]:.3f}")
        a = np.gradient(v, t)
        fig, axes = plt.subplots(3,1,figsize=(8,7))
        axes[0].plot(t,x); axes[0].set_ylabel('x'); axes[0].grid(True,alpha=0.3)
        axes[1].plot(t,v); axes[1].set_ylabel('v'); axes[1].grid(True,alpha=0.3)
        axes[2].plot(t,a); axes[2].set_ylabel('a'); axes[2].set_xlabel('t'); axes[2].grid(True,alpha=0.3)
        plt.tight_layout(); plt.savefig('bvp_steering_result.png',dpi=140); print("Plot saved: bvp_steering_result.png")
        plt.show()
    else:
        print("✗ Steering failed")
    print("="*70)

if __name__ == '__main__':
    test_bvp_steering()
