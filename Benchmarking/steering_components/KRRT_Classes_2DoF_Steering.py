# Moved to steering_components: Steering + KRRT classes
import cupy as cp
import numpy as np
from cupyx.scipy.spatial import KDTree
from typing import List, Tuple, Annotated, TypeAlias, Optional
from cupy.typing import NDArray
try:
    from ompl import base as ob
    from ompl import control as oc
    OMPL_AVAILABLE = True
except ImportError:
    OMPL_AVAILABLE = False
try:
    from scipy.optimize import minimize, LinearConstraint, Bounds
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
PositionVector2D: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
VelocityVector2D: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
KinodynamicState2D: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
ControlInput2D: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
class RobotState2D:
    DOF: int = 2
    MAX_POSITIONS: PositionVector2D = cp.array([10.0,10.0])
    MAX_VELOCITIES: VelocityVector2D = cp.array([2.0,2.0])
    MAX_ACCELERATIONS: PositionVector2D = cp.array([3.0,3.0])
    MIN_POSITIONS: PositionVector2D = cp.array([-10.0,-10.0])
    def __init__(self, positions: PositionVector2D, velocities: VelocityVector2D) -> None:
        self.positions = positions.copy(); self.velocities = velocities.copy()
    def positions_to_vector(self) -> PositionVector2D: return self.positions
    def velocities_to_vector(self) -> VelocityVector2D: return self.velocities
    def to_vector(self) -> KinodynamicState2D: return cp.concatenate((self.positions_to_vector(), self.velocities_to_vector()))
    def copy(self) -> 'RobotState2D': return RobotState2D(self.positions.copy(), self.velocities.copy())
class ControlProfile2D:
    def __init__(self,duration:float,accelerations:List[NDArray[cp.float64]],time_steps:List[float]) -> None:
        self.duration=duration; self.accelerations=accelerations; self.time_steps=time_steps
    def get_duration(self)->float: return self.duration
    def get_accelerations(self)->List[NDArray[cp.float64]]: return self.accelerations
    def get_time_steps(self)->List[float]: return self.time_steps
    def copy(self)->'ControlProfile2D': return ControlProfile2D(self.duration,[a.copy() for a in self.accelerations],self.time_steps.copy())
class KinodynamicRRTNode2D:
    def __init__(self,state:RobotState2D,cost:float,control_profile:Optional[ControlProfile2D]=None)->None:
        self.robot_state=state; self.parent=None; self.cost=cost; self.control_profile=control_profile
    def get_parent(self)->'KinodynamicRRTNode2D': return self.parent
    def get_state(self)->RobotState2D: return self.robot_state
    def get_cost(self)->float: return self.cost
    def get_control_profile(self)->Optional[ControlProfile2D]: return self.control_profile
    def set_cost(self,cost:float)->None: self.cost=cost
    def set_parent(self,parent:'KinodynamicRRTNode2D')->None: self.parent=parent
    def set_control_profile(self,profile:ControlProfile2D)->None: self.control_profile=profile
    def to_vector(self)->KinodynamicState2D: return self.robot_state.to_vector()
MAX_STEERING_TIME: float = 2.0
NUM_CONTROL_SEGMENTS: int = 5
class OMPLSteeringFunction:
    def __init__(self):
        if not OMPL_AVAILABLE: raise ImportError("OMPL not available")
        self.state_space = ob.CompoundStateSpace(); position_space=ob.RealVectorStateSpace(2); pos_bounds=ob.RealVectorBounds(2)
        pos_bounds.setLow(0,float(cp.asnumpy(RobotState2D.MIN_POSITIONS[0]))); pos_bounds.setHigh(0,float(cp.asnumpy(RobotState2D.MAX_POSITIONS[0])))
        pos_bounds.setLow(1,float(cp.asnumpy(RobotState2D.MIN_POSITIONS[1]))); pos_bounds.setHigh(1,float(cp.asnumpy(RobotState2D.MAX_POSITIONS[1])))
        position_space.setBounds(pos_bounds); velocity_space=ob.RealVectorStateSpace(2); vel_bounds=ob.RealVectorBounds(2)
        vel_bounds.setLow(0,float(-cp.asnumpy(RobotState2D.MAX_VELOCITIES[0]))); vel_bounds.setHigh(0,float(cp.asnumpy(RobotState2D.MAX_VELOCITIES[0])))
        vel_bounds.setLow(1,float(-cp.asnumpy(RobotState2D.MAX_VELOCITIES[1]))); vel_bounds.setHigh(1,float(cp.asnumpy(RobotState2D.MAX_VELOCITIES[1])))
        velocity_space.setBounds(vel_bounds); self.state_space.addSubspace(position_space,1.0); self.state_space.addSubspace(velocity_space,0.5)
        self.control_space=oc.RealVectorControlSpace(self.state_space,2); control_bounds=ob.RealVectorBounds(2)
        control_bounds.setLow(0,float(-cp.asnumpy(RobotState2D.MAX_ACCELERATIONS[0]))); control_bounds.setHigh(0,float(cp.asnumpy(RobotState2D.MAX_ACCELERATIONS[0])))
        control_bounds.setLow(1,float(-cp.asnumpy(RobotState2D.MAX_ACCELERATIONS[1]))); control_bounds.setHigh(1,float(cp.asnumpy(RobotState2D.MAX_ACCELERATIONS[1])))
        self.control_space.setBounds(control_bounds); self.si=oc.SpaceInformation(self.state_space,self.control_space)
        self.si.setStatePropagator(oc.StatePropagatorFn(self._propagate)); self.si.setPropagationStepSize(0.05); self.si.setMinMaxControlDuration(1,20); self.si.setup()
    def _propagate(self,start,control,duration,state):
        x,y = start[0][0],start[0][1]; vx,vy=start[1][0],start[1][1]; ax,ay=control[0],control[1]; dt=duration
        vx_new = vx + ax*dt; vy_new = vy + ay*dt
        vx_new = max(min(vx_new,float(cp.asnumpy(RobotState2D.MAX_VELOCITIES[0]))),float(-cp.asnumpy(RobotState2D.MAX_VELOCITIES[0])))
        vy_new = max(min(vy_new,float(cp.asnumpy(RobotState2D.MAX_VELOCITIES[1]))),float(-cp.asnumpy(RobotState2D.MAX_VELOCITIES[1])))
        x_new = x + 0.5*(vx+vx_new)*dt; y_new = y + 0.5*(vy+vy_new)*dt
        x_new = max(min(x_new,float(cp.asnumpy(RobotState2D.MAX_POSITIONS[0]))),float(cp.asnumpy(RobotState2D.MIN_POSITIONS[0])))
        y_new = max(min(y_new,float(cp.asnumpy(RobotState2D.MAX_POSITIONS[1]))),float(cp.asnumpy(RobotState2D.MIN_POSITIONS[1])))
        state[0][0],state[0][1]=x_new,y_new; state[1][0],state[1][1]=vx_new,vy_new
    def steer(self,initial_state:RobotState2D,target_state:RobotState2D,max_time:float=MAX_STEERING_TIME)->Optional[Tuple[ControlProfile2D,RobotState2D]]:
        start=self.si.allocState(); goal=self.si.allocState()
        start[0][0],start[0][1]=float(cp.asnumpy(initial_state.positions_to_vector()[0])), float(cp.asnumpy(initial_state.positions_to_vector()[1]))
        start[1][0],start[1][1]=float(cp.asnumpy(initial_state.velocities_to_vector()[0])), float(cp.asnumpy(initial_state.velocities_to_vector()[1]))
        goal[0][0],goal[0][1]=float(cp.asnumpy(target_state.positions_to_vector()[0])), float(cp.asnumpy(target_state.positions_to_vector()[1]))
        goal[1][0],goal[1][1]=float(cp.asnumpy(target_state.velocities_to_vector()[0])), float(cp.asnumpy(target_state.velocities_to_vector()[1]))
        control_sampler=self.control_space.allocControlSampler(); control=self.si.allocControl(); num_samples=20
        best_distance=float('inf'); best_control_seq=None; best_final_state=None; best_duration=0.0
        for _ in range(num_samples):
            control_sampler.sample(control)
            for duration_steps in range(5,25,2):
                duration=duration_steps*self.si.getPropagationStepSize()
                if duration>max_time: continue
                result=self.si.allocState(); self.si.propagate(start,control,duration,result)
                dist=self.state_space.distance(result,goal)
                if dist < best_distance:
                    best_distance=dist; best_control_seq=[control[0],control[1]]; best_final_state=result; best_duration=duration
        if best_control_seq is None: return None
        final_pos=cp.array([best_final_state[0][0],best_final_state[0][1]],dtype=cp.float64); final_vel=cp.array([best_final_state[1][0],best_final_state[1][1]],dtype=cp.float64)
        final_state=RobotState2D(final_pos,final_vel); accel=cp.array(best_control_seq,dtype=cp.float64); profile=ControlProfile2D(best_duration,[accel],[best_duration]); return profile,final_state

def steer_optimal_2d_ompl(initial_state:RobotState2D,target_state:RobotState2D,max_time:float=MAX_STEERING_TIME)->Optional[Tuple[ControlProfile2D,RobotState2D]]:
    if not OMPL_AVAILABLE: return steer_optimal_2d_fallback(initial_state,target_state,max_time)
    try:
        if not hasattr(steer_optimal_2d_ompl,'_ompl_steerer'): steer_optimal_2d_ompl._ompl_steerer=OMPLSteeringFunction()
        return steer_optimal_2d_ompl._ompl_steerer.steer(initial_state,target_state,max_time)
    except Exception: return steer_optimal_2d_fallback(initial_state,target_state,max_time)

def steer_optimal_2d_qp(initial_state:RobotState2D,target_state:RobotState2D,max_time:float=MAX_STEERING_TIME)->Optional[Tuple[ControlProfile2D,RobotState2D]]:
    if not SCIPY_AVAILABLE: return steer_optimal_2d_fallback(initial_state,target_state,max_time)
    N=NUM_CONTROL_SEGMENTS; DOF=RobotState2D.DOF
    x0=cp.asnumpy(initial_state.positions_to_vector()); v0=cp.asnumpy(initial_state.velocities_to_vector())
    x_target=cp.asnumpy(target_state.positions_to_vector()); v_target=cp.asnumpy(target_state.velocities_to_vector())
    a_max=cp.asnumpy(RobotState2D.MAX_ACCELERATIONS); n_vars=1+DOF*N; control_weight=0.01
    def objective(x): T=x[0]; accelerations=x[1:].reshape(N,DOF); return T + control_weight*np.sum(accelerations**2)
    def objective_grad(x): grad=np.zeros(n_vars); grad[0]=1.0; grad[1:]=2*control_weight*x[1:]; return grad
    def constraint_final_state(x):
        T=x[0]; dt=T/N; accelerations=x[1:].reshape(N,DOF); pos=x0.copy(); vel=v0.copy()
        for i in range(N): a=accelerations[i]; vel_new=vel + a*dt; pos_new=pos + vel*dt + 0.5*a*dt**2; vel=vel_new; pos=pos_new
        return np.concatenate([pos - x_target, vel - v_target])
    def constraint_final_state_jac(x):
        jac=np.zeros((2*DOF,n_vars)); eps=1e-7; f=constraint_final_state(x)
        for j in range(n_vars): x_plus=x.copy(); x_plus[j]+=eps; f_plus=constraint_final_state(x_plus); jac[:,j]=(f_plus-f)/eps
        return jac
    from scipy.optimize import NonlinearConstraint
    bounds=Bounds(lb=np.concatenate([[0.1], -np.tile(a_max,N)]), ub=np.concatenate([[max_time], np.tile(a_max,N)]))
    nlc=NonlinearConstraint(constraint_final_state, lb=-1e-3*np.ones(2*DOF), ub=1e-3*np.ones(2*DOF), jac=constraint_final_state_jac)
    x_init=np.concatenate([[max_time/2], np.zeros(DOF*N)])
    result=minimize(objective,x_init,method='SLSQP',jac=objective_grad,bounds=bounds,constraints=[nlc],options={'maxiter':200,'ftol':1e-6})
    if not result.success or result.x[0] < 0.1: return steer_optimal_2d_fallback(initial_state,target_state,max_time)
    T_opt=result.x[0]; dt=T_opt/N; accelerations_opt=result.x[1:].reshape(N,DOF)
    accel_arrays=[cp.array(acc,dtype=cp.float64) for acc in accelerations_opt]; time_steps=[dt]*N
    pos=cp.array(x0,dtype=cp.float64); vel=cp.array(v0,dtype=cp.float64)
    for i in range(N): a=accel_arrays[i]; vel=vel+a*dt; pos=pos+vel*dt - 0.5*a*dt**2
    final_state=RobotState2D(pos,vel); profile=ControlProfile2D(T_opt,accel_arrays,time_steps); return profile,final_state

def steer_optimal_2d_fallback(initial_state:RobotState2D,target_state:RobotState2D,max_time:float=MAX_STEERING_TIME)->Optional[Tuple[ControlProfile2D,RobotState2D]]:
    DOF=RobotState2D.DOF; x0=initial_state.positions_to_vector(); v0=initial_state.velocities_to_vector(); x_target=target_state.positions_to_vector(); v_target=target_state.velocities_to_vector()
    best_profile=None; best_final_state=None; best_error=float('inf')
    for T in cp.linspace(0.1, max_time, 15):
        T = float(T)
        valid = True
        final_pos = cp.zeros(DOF)
        final_vel = cp.zeros(DOF)
        accels_dim = []
        for i in range(DOF):
            dx = x_target[i] - x0[i]
            dv = v_target[i] - v0[i]
            a_max = RobotState2D.MAX_ACCELERATIONS[i]
            # Simple two-phase heuristic split
            t1 = T / 2.0
            t2 = T - t1
            a1 = a_max if dx > 0 else -a_max
            v_mid = v0[i] + a1 * t1
            x_mid = x0[i] + v0[i] * t1 + 0.5 * a1 * t1**2
            if t2 > 0:
                a2 = (v_target[i] - v_mid) / t2
                if abs(a2) > a_max:
                    valid = False
                    break
                v_final = v_mid + a2 * t2
                x_final = x_mid + v_mid * t2 + 0.5 * a2 * t2**2
            else:
                v_final = v_mid
                x_final = x_mid
                a2 = 0.0
            # Constraint checks
            if (abs(v_final) > RobotState2D.MAX_VELOCITIES[i] or
                x_final > RobotState2D.MAX_POSITIONS[i] or
                x_final < RobotState2D.MIN_POSITIONS[i]):
                valid = False
                break
            final_pos[i] = x_final
            final_vel[i] = v_final
            accels_dim.append((a1, a2, t1))
        if valid:
            error = cp.linalg.norm(final_pos - x_target) + 0.5 * cp.linalg.norm(final_vel - v_target)
            if error < best_error:
                best_error = float(error)
                accel_arrays = [
                    cp.array([accels_dim[j][0] for j in range(DOF)], dtype=cp.float64),
                    cp.array([accels_dim[j][1] for j in range(DOF)], dtype=cp.float64)
                ]
                time_steps = [accels_dim[0][2], T - accels_dim[0][2]]
                best_profile = ControlProfile2D(T, accel_arrays, time_steps)
                best_final_state = RobotState2D(final_pos.copy(), final_vel.copy())
    return (best_profile,best_final_state) if best_profile is not None else None

def steer_optimal_2d(initial_state:RobotState2D,target_state:RobotState2D,max_time:float=MAX_STEERING_TIME)->Optional[Tuple[ControlProfile2D,RobotState2D]]:
    if OMPL_AVAILABLE:
        r = steer_optimal_2d_ompl(initial_state,target_state,max_time)
        if r is not None: return r
    if SCIPY_AVAILABLE:
        r = steer_optimal_2d_qp(initial_state,target_state,max_time)
        if r is not None: return r
    return steer_optimal_2d_fallback(initial_state,target_state,max_time)

def apply_control_profile(initial_state:RobotState2D,profile:ControlProfile2D)->RobotState2D:
    state=initial_state.copy()
    for accel,dt in zip(profile.get_accelerations(),profile.get_time_steps()):
        v0=state.velocities_to_vector().copy(); x0=state.positions_to_vector().copy(); v_new=v0+accel*dt; x_new=x0+v0*dt+0.5*accel*dt*dt; state.velocities[:]=v_new; state.positions[:]=x_new
    return state
class KRRT_KDTree2D:
    def __init__(self,nodes:List[KinodynamicRRTNode2D])->None:
        self.nodes=nodes; states=cp.array([node.to_vector() for node in nodes]); self._kd_tree=KDTree(states)
    def query(self,sample_state:KinodynamicRRTNode2D,k:int=1)->List[Tuple[KinodynamicRRTNode2D,float]]:
        sample_vector=sample_state.robot_state.to_vector().reshape(1,-1); dist,indexes=self._kd_tree.query(sample_vector,k=k)
        if k==1: return [(self.nodes[int(indexes[0])], float(dist[0]))]
        else: return [(self.nodes[i], float(dist[0][j])) for j,i in enumerate(indexes[0])]
    def query_ball_point(self,sample_state:KinodynamicRRTNode2D,r:float)->List[KinodynamicRRTNode2D]:
        sample_vector=sample_state.robot_state.to_vector().reshape(1,-1); indices=self._kd_tree.query_ball_point(sample_vector,r); return [self.nodes[i] for i in indices[0]]
    def add_node(self,node:KinodynamicRRTNode2D)->None:
        self.nodes.append(node); states=cp.array([n.to_vector() for n in self.nodes]); self._kd_tree=KDTree(states)
class KRRT_Star_Calculator2D:
    def __init__(self,initial_positions:PositionVector2D,goal_positions:PositionVector2D,initial_velocities:VelocityVector2D=cp.zeros(2),goal_velocities:VelocityVector2D=cp.zeros(2))->None:
        self.start_node=KinodynamicRRTNode2D(RobotState2D(initial_positions,initial_velocities),0.0)
        self.goal_node=KinodynamicRRTNode2D(RobotState2D(goal_positions,goal_velocities),float('inf'))
        self.nodes=KRRT_KDTree2D([self.start_node]); self.N=1
    @staticmethod
    def from_states(initial_state:RobotState2D,goal_state:RobotState2D)->'KRRT_Star_Calculator2D':
        return KRRT_Star_Calculator2D(initial_state.positions_to_vector(),goal_state.positions_to_vector(),initial_state.velocities_to_vector(),goal_state.velocities_to_vector())
    def get_nearest_node(self,sample_state:RobotState2D)->KinodynamicRRTNode2D:
        return self.nodes.query(KinodynamicRRTNode2D(sample_state,0.0),k=1)[0][0]
    def get_connection_radius(self)->float:
        gamma_rrt_star=2.0*(1+1.0/(2*RobotState2D.DOF))**(1.0/(2*RobotState2D.DOF)); return float(gamma_rrt_star*((cp.log(self.N)/self.N)**(1.0/(2*RobotState2D.DOF))))
    def sample_random_state(self)->RobotState2D:
        positions=RobotState2D.MIN_POSITIONS + (RobotState2D.MAX_POSITIONS - RobotState2D.MIN_POSITIONS) * cp.random.uniform(0.0,1.0,size=(RobotState2D.DOF,),dtype=cp.float64)
        velocities=RobotState2D.MAX_VELOCITIES * cp.random.uniform(-1.0,1.0,size=(RobotState2D.DOF,),dtype=cp.float64)
        return RobotState2D(positions,velocities)
    def sample_goal_biased_state(self,bias_probability:float=0.1)->RobotState2D:
        return self.goal_node.get_state().copy() if cp.random.uniform() < bias_probability else self.sample_random_state()
    @staticmethod
    def check_collision(state1:RobotState2D,state2:RobotState2D)->bool: return False
    def check_path_collision(self,initial_state:RobotState2D,profile:ControlProfile2D,num_checks:int=10)->bool:
        state=initial_state.copy()
        for accel,dt in zip(profile.get_accelerations(),profile.get_time_steps()):
            segment_checks=max(1,int(num_checks*dt/profile.get_duration())); dt_check=dt/segment_checks
            for _ in range(segment_checks):
                v0=state.velocities_to_vector().copy(); x0=state.positions_to_vector().copy(); v_new=v0+accel*dt_check; x_new=x0+v0*dt_check+0.5*accel*dt_check**2
                state.velocities[:]=v_new; state.positions[:]=x_new
                if self.check_collision(initial_state,state): return True
        return False
    def steer_towards(self,from_node:KinodynamicRRTNode2D,target_state:RobotState2D)->Optional[Tuple[KinodynamicRRTNode2D,ControlProfile2D]]:
        result=steer_optimal_2d(from_node.get_state(),target_state,MAX_STEERING_TIME)
        if result is None: return None
        control_profile, final_state = result
        if self.check_path_collision(from_node.get_state(),control_profile): return None
        new_cost=from_node.get_cost()+control_profile.get_duration(); new_node=KinodynamicRRTNode2D(final_state,new_cost,control_profile); new_node.set_parent(from_node); return new_node,control_profile
    def add_node(self,node:KinodynamicRRTNode2D)->None:
        self.nodes.add_node(node); self.rewire(node); self.N+=1
    def rewire(self,new_node:KinodynamicRRTNode2D)->bool:
        neighboring_nodes=self.nodes.query_ball_point(new_node,self.get_connection_radius()); rewired=False
        for neighbor in neighboring_nodes:
            if neighbor == new_node or neighbor == self.start_node: continue
            result=steer_optimal_2d(new_node.get_state(),neighbor.get_state(),MAX_STEERING_TIME)
            if result is not None:
                control_profile, final_state = result; potential_cost = new_node.get_cost() + control_profile.get_duration()
                if potential_cost < neighbor.get_cost() and not self.check_path_collision(new_node.get_state(),control_profile):
                    neighbor.set_parent(new_node); neighbor.set_control_profile(control_profile); neighbor.set_cost(potential_cost); rewired=True
            result=steer_optimal_2d(neighbor.get_state(),new_node.get_state(),MAX_STEERING_TIME)
            if result is not None:
                control_profile, final_state = result; potential_cost = neighbor.get_cost() + control_profile.get_duration()
                if potential_cost < new_node.get_cost() and not self.check_path_collision(neighbor.get_state(),control_profile):
                    new_node.set_parent(neighbor); new_node.set_control_profile(control_profile); new_node.set_cost(potential_cost); rewired=True
        return rewired
    def get_path_and_controls(self,node:KinodynamicRRTNode2D)->Tuple[List[KinodynamicRRTNode2D],List[ControlProfile2D]]:
        path=[]; controls=[]; current=node
        while current is not None:
            path.append(current)
            if current.get_control_profile() is not None: controls.append(current.get_control_profile())
            current=current.get_parent()
        return path[::-1],controls[::-1]
    def check_goal_reached(self,node:KinodynamicRRTNode2D,position_tolerance:float=0.1,velocity_tolerance:float=0.1)->bool:
        pos_diff=cp.linalg.norm(node.get_state().positions_to_vector()-self.goal_node.get_state().positions_to_vector())
        vel_diff=cp.linalg.norm(node.get_state().velocities_to_vector()-self.goal_node.get_state().velocities_to_vector())
        return pos_diff < position_tolerance and vel_diff < velocity_tolerance
