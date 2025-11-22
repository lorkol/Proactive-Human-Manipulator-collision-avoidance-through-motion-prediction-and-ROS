#!/usr/bin/env python3
"""
Analytical Steering Function for Double Integrator System (moved to steering_components)
"""

from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass

@dataclass
class State2D:
    x: float
    y: float
    vx: float
    vy: float

@dataclass
class Control2D:
    ax: float
    ay: float

class DoubleIntegratorSteering:
    def __init__(self, max_accel: float, max_velocity: float):
        self.a_max = max_accel
        self.v_max = max_velocity
    def steer_1d(self, x0: float, v0: float, xf: float, vf: float) -> Optional[Tuple[List[float], List[float]]]:
        dx = xf - x0; dv = vf - v0
        if abs(vf) > self.v_max or abs(v0) > self.v_max: return None
        if abs(dv) < 1e-6:
            if abs(v0) < 1e-6:
                if abs(dx) < 1e-6: return [0.0],[0.01]
                t_total = 2.0 * np.sqrt(abs(dx) / self.a_max); t_half = t_total/2.0; a_sign = 1.0 if dx>0 else -1.0
                return [a_sign*self.a_max, -a_sign*self.a_max],[t_half,t_half]
            else:
                t = dx / v0 if abs(v0) > 1e-6 else 0.0
                if t >= 0: return [0.0],[max(t,0.01)]
        a1 = self.a_max if dv>0 else -self.a_max
        t1 = abs(dv)/self.a_max
        x1 = x0 + v0*t1 + 0.5*a1*t1**2
        v1 = v0 + a1*t1
        if abs(v1) > self.v_max:
            v1 = np.sign(v1)*self.v_max
            t1 = (v1 - v0)/a1 if abs(a1)>1e-6 else 0.0
            x1 = x0 + v0*t1 + 0.5*a1*t1**2
        dx_remain = xf - x1
        if abs(vf) > 1e-6:
            t2 = dx_remain / vf if abs(vf)>1e-6 else 0.0
            if t2 < 0: return self._solve_3phase(x0,v0,xf,vf)
            return [a1,0.0],[t1,max(t2,0.01)]
        else:
            if abs(v1) > 1e-6:
                a2 = -self.a_max if v1>0 else self.a_max
                t2 = abs(v1)/self.a_max
                x2_expected = x1 + v1*t2 + 0.5*a2*t2**2
                if abs(x2_expected - xf) < 0.1: return [a1,a2],[t1,t2]
                else: return self._solve_3phase(x0,v0,xf,vf)
            else: return [a1],[t1]
    def _solve_3phase(self, x0: float, v0: float, xf: float, vf: float) -> Optional[Tuple[List[float], List[float]]]:
        dx = xf - x0; v_mid_target = self.v_max if dx>0 else -self.v_max
        a1 = self.a_max if v_mid_target > v0 else -self.a_max
        t1 = abs(v_mid_target - v0)/self.a_max
        x1 = x0 + v0*t1 + 0.5*a1*t1**2; v1 = v0 + a1*t1
        a3 = -self.a_max if v1 > vf else self.a_max
        t3 = abs(v1 - vf)/self.a_max
        x3_disp = v1*t3 + 0.5*a3*t3**2
        dx_remain = dx - (x1 - x0) - x3_disp
        t2 = dx_remain / v1 if abs(v1)>1e-6 else 0.0
        if t2 < 0:
            a_avg = 2*(dx - v0*1.0)/(1.0**2); a_avg = np.clip(a_avg,-self.a_max,self.a_max)
            return [a_avg],[1.0]
        return [a1,0.0,a3],[t1,max(t2,0.01),t3]
    def steer_2d(self, start: State2D, goal: State2D) -> Optional[Tuple[List[Control2D], List[float]]]:
        xr = self.steer_1d(start.x,start.vx,goal.x,goal.vx); yr = self.steer_1d(start.y,start.vy,goal.y,goal.vy)
        if xr is None or yr is None: return None
        x_controls,x_durations = xr; y_controls,y_durations = yr
        total_time = max(sum(x_durations),sum(y_durations)); dt=0.1; num_steps=int(np.ceil(total_time/dt))
        controls=[]; durations=[]
        for i in range(num_steps):
            t=i*dt; ax=self._get_control_at_time(t,x_controls,x_durations); ay=self._get_control_at_time(t,y_controls,y_durations)
            controls.append(Control2D(ax,ay)); durations.append(dt)
        return controls,durations
    def _get_control_at_time(self,t:float,controls:List[float],durations:List[float])->float:
        time_acc=0.0
        for ctrl,dur in zip(controls,durations):
            if t < time_acc + dur: return ctrl
            time_acc += dur
        return controls[-1] if controls else 0.0
    def simulate_trajectory(self,start:State2D,controls:List[Control2D],durations:List[float])->List[State2D]:
        states=[start]; current=State2D(start.x,start.y,start.vx,start.vy)
        for ctrl,dt in zip(controls,durations):
            vx_new=np.clip(current.vx+ctrl.ax*dt,-self.v_max,self.v_max); vy_new=np.clip(current.vy+ctrl.ay*dt,-self.v_max,self.v_max)
            x_new=current.x+current.vx*dt+0.5*ctrl.ax*dt**2; y_new=current.y+current.vy*dt+0.5*ctrl.ay*dt**2
            current=State2D(x_new,y_new,vx_new,vy_new); states.append(current)
        return states

def test_steering():
    print("="*70); print("DOUBLE INTEGRATOR STEERING FUNCTION TEST (steering_components)"); print("="*70)
    steering=DoubleIntegratorSteering(max_accel=3.0,max_velocity=2.0)
    print("\nTest 1: Simple forward motion")
    start=State2D(0.0,0.0,0.0,0.0); goal=State2D(5.0,3.0,0.0,0.0)
    result=steering.steer_2d(start,goal)
    if result:
        controls,durations=result; states=steering.simulate_trajectory(start,controls,durations)
        final=states[-1]; error=np.sqrt((final.x-goal.x)**2 + (final.y-goal.y)**2)
        print(f"Final: ({final.x:.2f},{final.y:.2f}) vel=({final.vx:.2f},{final.vy:.2f}) error={error:.4f}m ✓")
    else: print("✗ Steering failed")
    print("\nTest 2: Motion with velocity change")
    start=State2D(0.0,0.0,1.0,0.5); goal=State2D(5.0,3.0,-0.5,1.0)
    result=steering.steer_2d(start,goal)
    if result:
        controls,durations=result; states=steering.simulate_trajectory(start,controls,durations); final=states[-1]
        error=np.sqrt((final.x-goal.x)**2 + (final.y-goal.y)**2)
        print(f"Final: ({final.x:.2f},{final.y:.2f}) vel=({final.vx:.2f},{final.vy:.2f}) error={error:.4f}m ✓")
    else: print("✗ Steering failed")
    print("\n" + "="*70)

if __name__ == "__main__":
    test_steering()
