#!/usr/bin/env python3
"""OMPL Kinodynamic RRT with Analytical Steering (moved to steering_components)."""
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from ompl import base as ob
from ompl import control as oc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

class RobotConfig:
    MIN_X, MAX_X = -10.0, 10.0
    MIN_Y, MAX_Y = -10.0, 10.0
    MAX_VX, MAX_VY = 2.0, 2.0
    MAX_AX, MAX_AY = 3.0, 3.0
    PROPAGATION_STEP_SIZE = 0.05
    MIN_CONTROL_DURATION, MAX_CONTROL_DURATION = 5, 20

class AnalyticalSteering:
    def __init__(self, max_accel: float, max_velocity: float):
        self.a_max = max_accel; self.v_max = max_velocity
    def steer_1d(self, x0: float, v0: float, xf: float, vf: float) -> Optional[Tuple[List[float], List[float]]]:
        dx, dv = xf-x0, vf-v0
        if abs(vf) > self.v_max or abs(v0) > self.v_max: return None
        if abs(dv) < 1e-6:
            if abs(v0) < 1e-6:
                if abs(dx) < 1e-6: return [0.0],[0.01]
                t_half = np.sqrt(abs(dx)/self.a_max); a_sign = 1.0 if dx>0 else -1.0
                return [a_sign*self.a_max, -a_sign*self.a_max],[t_half,t_half]
            else:
                t = dx/v0 if abs(v0)>1e-6 else 0.01
                if t >= 0: return [0.0],[max(t,0.01)]
        a1 = self.a_max if dv>0 else -self.a_max
        t1 = min(abs(dv)/self.a_max,2.0)
        x1 = x0 + v0*t1 + 0.5*a1*t1**2
        v1 = np.clip(v0 + a1*t1, -self.v_max, self.v_max)
        dx_remain = xf - x1
        if abs(vf) > 1e-6:
            t2 = abs(dx_remain / vf) if abs(vf)>1e-6 else 0.01
            return [a1,0.0],[t1,max(t2,0.01)]
        else:
            if abs(v1) > 1e-6:
                a2 = -self.a_max if v1>0 else self.a_max
                t2 = abs(v1/self.a_max)
                return [a1,a2],[t1,t2]
            return [a1],[t1]
    def steer_2d(self, x0,y0,vx0,vy0, xf,yf,vxf,vyf) -> Optional[Tuple[List[Tuple[float,float]], List[float]]]:
        xr = self.steer_1d(x0,vx0,xf,vxf); yr = self.steer_1d(y0,vy0,yf,vyf)
        if xr is None or yr is None: return None
        x_controls,x_durations = xr; y_controls,y_durations = yr
        total_time = max(sum(x_durations),sum(y_durations)); dt=0.1; num_steps=max(int(np.ceil(total_time/dt)),1)
        controls=[]; durations=[]
        for i in range(num_steps):
            t=i*dt; ax=self._get_control_at_time(t,x_controls,x_durations); ay=self._get_control_at_time(t,y_controls,y_durations)
            controls.append((ax,ay)); durations.append(dt)
        return controls,durations
    def _get_control_at_time(self,t:float,controls:List[float],durations:List[float])->float:
        acc=0.0
        for ctrl,dur in zip(controls,durations):
            if t < acc + dur: return ctrl
            acc += dur
        return controls[-1] if controls else 0.0

class SteeringRRT:
    def __init__(self, si: Any, steering: AnalyticalSteering, obstacle_checker: Any):
        self.si = si; self.steering=steering; self.obstacle_checker=obstacle_checker; self.nodes=[]
    def add_node(self,state:Any,parent_idx:Optional[int],control:Optional[Tuple[float,float]]=None,duration:float=0.0)->int:
        idx=len(self.nodes); self.nodes.append((state,parent_idx,control,duration)); return idx
    def nearest_neighbor(self,state:Any)->int:
        min_dist,nearest=float('inf'),0
        for idx,(node_state,_,_,_) in enumerate(self.nodes):
            dx=state[0][0]-node_state[0][0]; dy=state[0][1]-node_state[0][1]
            dvx=state[1][0]-node_state[1][0]; dvy=state[1][1]-node_state[1][1]
            dist=np.sqrt(dx**2+dy**2+0.3*(dvx**2+dvy**2))
            if dist < min_dist: min_dist,nearest=dist,idx
        return nearest
    def steer_toward(self,from_state:Any,to_state:Any)->Optional[Tuple[Any,Tuple[float,float],float]]:
        x0,y0=from_state[0][0],from_state[0][1]; vx0,vy0=from_state[1][0],from_state[1][1]
        xf,yf=to_state[0][0],to_state[0][1]; vxf,vyf=to_state[1][0],to_state[1][1]
        result=self.steering.steer_2d(x0,y0,vx0,vy0,xf,yf,vxf,vyf)
        if result is None or len(result[0])==0: return None
        controls,durations=result; control,duration=controls[0],min(durations[0],0.5)
        new_state=self.si.allocState(); self._propagate_with_saturation(from_state,control,duration,new_state); return new_state,control,duration
    def _propagate_with_saturation(self,start:Any,control:Tuple[float,float],duration:float,state:Any)->None:
        x,y=start[0][0],start[0][1]; vx,vy=start[1][0],start[1][1]; ax,ay=control; dt=duration
        vx_un=vx+ax*dt
        if vx_un>RobotConfig.MAX_VX:
            vx_new=RobotConfig.MAX_VX
            t1=max(0.0,min((RobotConfig.MAX_VX - vx)/ax if abs(ax)>1e-10 else 0.0,dt))
            x_new=x+vx*t1+0.5*ax*t1**2+RobotConfig.MAX_VX*(dt-t1)
        elif vx_un < -RobotConfig.MAX_VX:
            vx_new=-RobotConfig.MAX_VX
            t1=max(0.0,min((-RobotConfig.MAX_VX - vx)/ax if abs(ax)>1e-10 else 0.0,dt))
            x_new=x+vx*t1+0.5*ax*t1**2 - RobotConfig.MAX_VX*(dt-t1)
        else:
            vx_new=vx_un; x_new=x+vx*dt+0.5*ax*dt**2
        vy_un=vy+ay*dt
        if vy_un>RobotConfig.MAX_VY:
            vy_new=RobotConfig.MAX_VY
            t1=max(0.0,min((RobotConfig.MAX_VY - vy)/ay if abs(ay)>1e-10 else 0.0,dt))
            y_new=y+vy*t1+0.5*ay*t1**2+RobotConfig.MAX_VY*(dt-t1)
        elif vy_un < -RobotConfig.MAX_VY:
            vy_new=-RobotConfig.MAX_VY
            t1=max(0.0,min((-RobotConfig.MAX_VY - vy)/ay if abs(ay)>1e-10 else 0.0,dt))
            y_new=y+vy*t1+0.5*ay*t1**2 - RobotConfig.MAX_VY*(dt-t1)
        else:
            vy_new=vy_un; y_new=y+vy*dt+0.5*ay*dt**2
        state[0][0]=np.clip(x_new,RobotConfig.MIN_X,RobotConfig.MAX_X); state[0][1]=np.clip(y_new,RobotConfig.MIN_Y,RobotConfig.MAX_Y)
        state[1][0],state[1][1]=vx_new,vy_new
    def is_collision_free_path(self,from_state:Any,to_state:Any,num_checks:int=10)->bool:
        for i in range(num_checks+1):
            a=i/num_checks; test=self.si.allocState()
            test[0][0]=(1-a)*from_state[0][0]+a*to_state[0][0]; test[0][1]=(1-a)*from_state[0][1]+a*to_state[0][1]
            test[1][0]=(1-a)*from_state[1][0]+a*to_state[1][0]; test[1][1]=(1-a)*from_state[1][1]+a*to_state[1][1]
            if not self.obstacle_checker.is_valid_state(test): return False
        return True
    def is_goal_reached(self,state:Any,goal_state:Any,threshold:float=0.5)->bool:
        dx=state[0][0]-goal_state[0][0]; dy=state[0][1]-goal_state[0][1]; dvx=state[1][0]-goal_state[1][0]; dvy=state[1][1]-goal_state[1][1]
        return np.sqrt(dx**2+dy**2+0.3*(dvx**2+dvy**2)) < threshold
    def plan(self,start_state:Any,goal_state:Any,max_iterations:int=5000,goal_bias:float=0.1)->Optional[List[int]]:
        self.nodes=[]; self.add_node(start_state,None); sampler=self.si.allocStateSampler()
        for it in range(max_iterations):
            rand = goal_state if np.random.random() < goal_bias else self.si.allocState(); sampler.sampleUniform(rand) if rand is not goal_state else None
            nearest_idx=self.nearest_neighbor(rand); nearest_state=self.nodes[nearest_idx][0]
            steer_res=self.steer_toward(nearest_state,rand)
            if steer_res is None: continue
            new_state,control,duration=steer_res
            if not self.obstacle_checker.is_valid_state(new_state): continue
            if not self.is_collision_free_path(nearest_state,new_state): continue
            new_idx=self.add_node(new_state,nearest_idx,control,duration)
            if self.is_goal_reached(new_state,goal_state):
                print(f"✓ Goal reached after {it+1} iterations!"); return self.extract_path(new_idx)
            if (it+1)%500==0: print(f"  Iteration {it+1}/{max_iterations}, tree size: {len(self.nodes)}")
        print("✗ Max iterations reached"); return None
    def extract_path(self,goal_idx:int)->List[int]:
        path=[]; idx=goal_idx
        while idx is not None:
            path.append(idx); _,parent_idx,_,_ = self.nodes[idx]; idx=parent_idx
        path.reverse(); return path

class ObstacleChecker:
    def __init__(self)->None:
        self.obstacles: List[Tuple[float,float,float]]=[(2.0,2.0,1.0),(5.0,-2.0,1.2),(-3.0,4.0,0.8)]
    def is_valid_state(self,state:Any)->bool:
        x,y=state[0][0],state[0][1]
        for ox,oy,r in self.obstacles:
            if np.sqrt((x-ox)**2+(y-oy)**2) < r: return False
        return True

def propagate(start:Any,control:Any,duration:float,state:Any)->None:
    x,y=start[0][0],start[0][1]; vx,vy=start[1][0],start[1][1]; ax,ay=control[0],control[1]; dt=duration
    vx_un=vx+ax*dt
    if vx_un>RobotConfig.MAX_VX:
        vx_new=RobotConfig.MAX_VX; t1=max(0.0,min((RobotConfig.MAX_VX - vx)/ax if abs(ax)>1e-10 else 0.0,dt)); x_new=x+vx*t1+0.5*ax*t1**2+RobotConfig.MAX_VX*(dt-t1)
    elif vx_un < -RobotConfig.MAX_VX:
        vx_new=-RobotConfig.MAX_VX; t1=max(0.0,min((-RobotConfig.MAX_VX - vx)/ax if abs(ax)>1e-10 else 0.0,dt)); x_new=x+vx*t1+0.5*ax*t1**2 - RobotConfig.MAX_VX*(dt-t1)
    else:
        vx_new=vx_un; x_new=x+vx*dt+0.5*ax*dt**2
    vy_un=vy+ay*dt
    if vy_un>RobotConfig.MAX_VY:
        vy_new=RobotConfig.MAX_VY; t1=max(0.0,min((RobotConfig.MAX_VY - vy)/ay if abs(ay)>1e-10 else 0.0,dt)); y_new=y+vy*t1+0.5*ay*t1**2+RobotConfig.MAX_VY*(dt-t1)
    elif vy_un < -RobotConfig.MAX_VY:
        vy_new=-RobotConfig.MAX_VY; t1=max(0.0,min((-RobotConfig.MAX_VY - vy)/ay if abs(ay)>1e-10 else 0.0,dt)); y_new=y+vy*t1+0.5*ay*t1**2 - RobotConfig.MAX_VY*(dt-t1)
    else:
        vy_new=vy_un; y_new=y+vy*dt+0.5*ay*dt**2
    state[0][0]=np.clip(x_new,RobotConfig.MIN_X,RobotConfig.MAX_X); state[0][1]=np.clip(y_new,RobotConfig.MIN_Y,RobotConfig.MAX_Y)
    state[1][0],state[1][1]=vx_new,vy_new

def create_space_information()->Tuple[Any,ObstacleChecker]:
    state_space=ob.CompoundStateSpace(); pos_space=ob.RealVectorStateSpace(2); pos_bounds=ob.RealVectorBounds(2)
    pos_bounds.setLow(0,RobotConfig.MIN_X); pos_bounds.setHigh(0,RobotConfig.MAX_X); pos_bounds.setLow(1,RobotConfig.MIN_Y); pos_bounds.setHigh(1,RobotConfig.MAX_Y); pos_space.setBounds(pos_bounds)
    vel_space=ob.RealVectorStateSpace(2); vel_bounds=ob.RealVectorBounds(2)
    vel_bounds.setLow(0,-RobotConfig.MAX_VX); vel_bounds.setHigh(0,RobotConfig.MAX_VX); vel_bounds.setLow(1,-RobotConfig.MAX_VY); vel_bounds.setHigh(1,RobotConfig.MAX_VY); vel_space.setBounds(vel_bounds)
    state_space.addSubspace(pos_space,1.0); state_space.addSubspace(vel_space,0.3)
    control_space=oc.RealVectorControlSpace(state_space,2); cb=ob.RealVectorBounds(2)
    cb.setLow(0,-RobotConfig.MAX_AX); cb.setHigh(0,RobotConfig.MAX_AX); cb.setLow(1,-RobotConfig.MAX_AY); cb.setHigh(1,RobotConfig.MAX_AY); control_space.setBounds(cb)
    si=oc.SpaceInformation(state_space,control_space); obstacle_checker=ObstacleChecker()
    si.setStateValidityChecker(ob.StateValidityCheckerFn(obstacle_checker.is_valid_state)); si.setStatePropagator(oc.StatePropagatorFn(propagate))
    si.setPropagationStepSize(RobotConfig.PROPAGATION_STEP_SIZE); si.setMinMaxControlDuration(RobotConfig.MIN_CONTROL_DURATION,RobotConfig.MAX_CONTROL_DURATION); si.setup(); return si,obstacle_checker

def create_state(si:Any,x:float,y:float,vx:float,vy:float)->Any:
    s=si.allocState(); s[0][0],s[0][1]=x,y; s[1][0],s[1][1]=vx,vy; return s

def plan_with_steering(start_pos:Tuple[float,float],goal_pos:Tuple[float,float],start_vel:Tuple[float,float]=(0.0,0.0),goal_vel:Tuple[float,float]=(0.0,0.0),planning_time:float=30.0)->Tuple[Optional[List[Tuple[float,float,float,float]]],Optional[List[Tuple[float,float,float]]],Dict[str,Any]]:
    print("="*70); print("OMPL KINODYNAMIC RRT WITH ANALYTICAL STEERING (steering_components)"); print("="*70)
    si, obstacle_checker = create_space_information(); start_state=create_state(si,start_pos[0],start_pos[1],start_vel[0],start_vel[1]); goal_state=create_state(si,goal_pos[0],goal_pos[1],goal_vel[0],goal_vel[1])
    steering=AnalyticalSteering(RobotConfig.MAX_AX,RobotConfig.MAX_VX); rrt=SteeringRRT(si,steering,obstacle_checker)
    print("Planning..."); start_time=time.time(); path_indices=rrt.plan(start_state,goal_state,max_iterations=5000,goal_bias=0.1); elapsed=time.time()-start_time
    stats={'solved':False,'planning_time':elapsed,'path_length':0,'num_states':0,'duration':0.0}
    if path_indices:
        solution_path=[]; solution_controls=[]
        for idx in path_indices:
            state,_,control,duration=rrt.nodes[idx]; x,y=state[0][0],state[0][1]; vx,vy=state[1][0],state[1][1]
            solution_path.append((x,y,vx,vy))
            if control is not None: solution_controls.append((control[0],control[1],duration)); stats['duration']+=duration
        path_length=sum(np.sqrt((solution_path[i+1][0]-solution_path[i][0])**2 + (solution_path[i+1][1]-solution_path[i][1])**2) for i in range(len(solution_path)-1))
        stats.update({'solved':True,'path_length':path_length,'num_states':len(solution_path)})
        return solution_path,solution_controls,stats
    return None,None,stats

def visualize_solution(solution_path:Optional[List],start_pos:Tuple,goal_pos:Tuple,obstacle_checker:ObstacleChecker,stats:Dict)->None:
    if solution_path is None: print('No solution'); return
    fig,ax=plt.subplots(1,1,figsize=(10,10)); ax.set_xlim(RobotConfig.MIN_X,RobotConfig.MAX_X); ax.set_ylim(RobotConfig.MIN_Y,RobotConfig.MAX_Y); ax.set_aspect('equal'); ax.grid(True,alpha=0.3)
    for ox,oy,r in obstacle_checker.obstacles: ax.add_patch(Circle((ox,oy),r,color='red',alpha=0.5))
    xs=[s[0] for s in solution_path]; ys=[s[1] for s in solution_path]; ax.plot(xs,ys,'b-',linewidth=2,alpha=0.7,label='Trajectory')
    ax.plot(start_pos[0],start_pos[1],'go',markersize=12,label='Start'); ax.plot(goal_pos[0],goal_pos[1],'r*',markersize=15,label='Goal'); ax.legend(); plt.tight_layout(); plt.savefig('kinodynamic_rrt_steering_result.png',dpi=150); print("Saved plot: kinodynamic_rrt_steering_result.png"); plt.show()

def main()->None:
    start_pos, start_vel = (-8.0,-8.0),(0.0,0.0); goal_pos, goal_vel = (8.0,7.0),(0.0,0.0)
    solution_path, solution_controls, stats = plan_with_steering(start_pos,goal_pos,start_vel,goal_vel,planning_time=30.0)
    if stats['solved']:
        si, obstacle_checker = create_space_information(); visualize_solution(solution_path,start_pos,goal_pos,obstacle_checker,stats)

if __name__ == '__main__':
    main()
