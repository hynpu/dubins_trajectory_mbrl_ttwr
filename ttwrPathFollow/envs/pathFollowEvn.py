import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ttwrPathFollow.ttwr_Simulator.vehicleModels import ttwrParams
from ttwrPathFollow.ttwr_Simulator.vehicleModels import ttwrModel
from ttwrPathFollow.ttwr_Simulator.vehicleModels import ttwrRender
import ttwrPathFollow.pathPlanner.dubinsPath as dubinsPath
from ttwrPathFollow.pathPlanner.utils.utils import *
import copy

class PathFollowEvn(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, controlMode=None):
        
        # define the constraints of the state and action spaces
        self.ttwr = ttwrModel.TtwrModel()
        self.renderModel = ttwrRender.VehicleRender()
        self.controlMode = controlMode

        self.x2_lim = ttwrParams.map_x_limit
        self.y2_lim = ttwrParams.map_y_limit
        self.theta2_lim = np.pi
        self.phi_lim = ttwrParams.jackKnifeAngle
        self.str_lim = ttwrParams.maxSteeringAngle
        self.turning_radius_limit = 15

        # define the time step and total steps
        self.t0 = 0.0
        self.t_final = 160.0
        self.dt = ttwrParams.dt
        self.max_step = int((self.t_final - self.t0)/self.dt) + 1
        self.step_idx = 0

        # define the look ahead sample index
        self.look_ahead = 0

        # fixed velocity for reverse parking
        self._velo = -2
        self._action = 0

        # the state is: [x2, y2, theta2, phi]
        # self.min_state = np.array([self.x2_lim, self.y2_lim, self.theta2_lim, self.phi_lim]) * (-1)
        # self.max_state = np.array([self.x2_lim, self.y2_lim, self.theta2_lim, self.phi_lim]) 

        self.min_state = np.array([10, 10, np.pi, np.pi]) * (-1)
        self.max_state = np.array([10, 10, np.pi, np.pi])

        # define observation space: trailer states
        self.observation_space = spaces.Box(low=self.min_state, high=self.max_state)

        # define action space: steering angle
        self.action_space = spaces.Box(low=(-self.str_lim), high=self.str_lim, shape=(1,))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self._ttwr_states = np.array([0, 0, 0, 0, 0, 0, 0])

        # define the error states weights, [theta1 err, theta2 err, trailer long err, trailer lat err]
        # only consider the (theta 2 err), and (trailer lateral error) for now
        self._err_states_weight = np.zeros((4, 4))
        self._err_states_weight[1, 1] = 1
        self._err_states_weight[3, 3] = 1

        self._action_weight = np.array([[0, 0], [0, 1]])

    def getPathPnts(self):
        return self.path_ref_pnts
    
    def reset(self, seed=None, options=None, trailer_state=None, target_state=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        print(self.ttwr.L3)
        # hard code the trailer and target states for now
        # trailer_state = np.array([25, 15, -np.pi/3, 0])
        # target_state = np.array([-25, -10, 0, 0])
        self._action = 0
        self.step_idx = 0
        # if trailer_state and target_state are given, use the given states
        if trailer_state is not None and target_state is not None:
            self.ttwr.reset(trailer_state, velo = -2)
            self._target_state = target_state
            self._get_obs_latErrMode()
            # set start and end states for dubins path
            # use deep copy to avoid the reference issue
            states_start = copy.deepcopy( self._ttwr_states[3:6] )
            states_end = copy.deepcopy( self._target_state[0:3] )
            # generate dubins path states along the path
            self.path_ref_pnts = dubinsPath.dubinsPathGenerator(states_start, states_end, self.turning_radius_limit, ttwrParams.path_res)
            # if (not self._trajectoryInsideBounds()):
            #     # raise error if the generated path is out of bounds
            #     raise ValueError("The generated path is out of bounds!")
            
        # if trailer_state and target_state are not given, generate random states for training
        else:
            while(True):
                # np random start position for the trailer, range is ttwrParams.map_x_limit and ttwrParams.map_y_limit
                x2_init = self.np_random.uniform(-self.x2_lim, self.x2_lim, size=1).item() # convert the numpy array to a scalar
                y2_init = self.np_random.uniform(-self.y2_lim, self.y2_lim, size=1).item()
                # np random start orientation for the trailer, range is [0, pi]
                theta2_init = self.np_random.uniform(-self.theta2_lim, self.theta2_lim, size=1).item()

                # np random end position for the trailer, range is ttwrParams.map_x_limit and ttwrParams.map_y_limit
                x2_end = self.np_random.uniform(-self.x2_lim, self.x2_lim, size=1).item() # convert the numpy array to a scalar
                y2_end = self.np_random.uniform(-self.y2_lim, self.y2_lim, size=1).item()
                # np random start orientation for the trailer, range is [0, pi]
                theta2_end = self.np_random.uniform(-self.theta2_lim, self.theta2_lim, size=1).item()
                self._target_state = np.array([x2_end, y2_end, theta2_end, 0])
                
                # Reset the ttwr states based on trailer states
                self.ttwr.reset(np.array([x2_init, y2_init, theta2_init, 0]), velo = -2)
                self._get_obs_latErrMode()

                # check the distance between start and end states
                dist_start_end = eucDistCompt(np.array([x2_init, y2_init]), np.array([x2_end, y2_end]))
                if dist_start_end < self.turning_radius_limit * 1.5:
                    continue

                # set start and end states for dubins path, use deep copy to avoid the referencing issue
                states_start = copy.deepcopy( self._ttwr_states[3:6] )
                states_end = copy.deepcopy( self._target_state[0:3] )
                
                self.path_ref_pnts = dubinsPath.dubinsPathGenerator(states_start, states_end, self.turning_radius_limit, ttwrParams.path_res)

                # if the generated path is inside bounds, break the loop and go to next section
                if (self._trajectoryInsideBounds()):
                    break

        # reset path related variables
        self.prev_trailer_ref_idx = 0
        self.prev_host_ref_idx = 0

        # if self.render_mode == "human":
        #     self._render_frame()

        if self.controlMode == 'lateralDeviation':
            envStates = self._get_error_latErrMode()
            info = self._get_info_latErrMode()
        elif self.controlMode == 'fullStateControl':
            envStates = self._ttwr_states
            info = {}
        else:        
            print("Invalid control mode!")
            quit()

        return envStates, info

    def _trajectoryInsideBounds(self):
        f_trajectoryInsideBounds = False
        # if all points on the self.path_ref_pnts is out of ttwrParams.map_x_limit, ttwrParams.map_y_limit, return True
        for i in range(len(self.path_ref_pnts[:,0])):
            if (self.path_ref_pnts[i,0] < -self.x2_lim or self.path_ref_pnts[i,0] > self.x2_lim or
                self.path_ref_pnts[i,1] < -self.y2_lim or self.path_ref_pnts[i,1] > self.y2_lim):
                f_trajectoryInsideBounds = False
                break
            else:
                f_trajectoryInsideBounds = True
        return f_trajectoryInsideBounds
    
    def close(self):
        ''' '''
        # self.renderModel.close()
    
    # generic step function for LQR, MPC, and RL
    def step(self, action):
        if self.controlMode == 'lateralDeviation':
            delta = action[0]
            accelation = 0
        elif self.controlMode == 'fullStateControl':
            accelation = action[0]
            delta = action[1]
        
        self.step_idx += 1
        self._action = delta.item()
        self.ttwr.move(accelation, self._action)
        self._ttwr_states = self.ttwr.state

        # this pattern is used to get the observation, reward, done, and info for lateral offset based control, no preview control
        if self.controlMode == 'lateralDeviation':
            # get the ttwr observation and target observation
            # self.observation = self._get_obs_latErrMode()

            self.deviate_err = self._get_error_latErrMode()

            reward, f_success, f_invalid = self._get_reward_latErrMode()

            # An episode is done if the agent has reached the target
            done = (f_success or f_invalid)

            info = self._get_info_latErrMode(reward, f_success, f_invalid)
            
            return self.deviate_err, reward, done, False, info
        
        elif self.controlMode == 'fullStateControl':
            self.deviate_err = self._get_error_latErrMode()
            f_parked = self._is_success()
            f_invalid = self._is_invalid()
            # done = (f_success or f_invalid)
            done = False
            info = {}
            return self._ttwr_states, 0, done, False, info
        else:
            print("Invalid control mode!")
            quit()


    ## below is the code for lateral deviation based TTWR control
    def _get_obs_latErrMode(self):
        self._ttwr_states = self.ttwr.state

        return {"ttwr": self._ttwr_states, "target": self._target_state}
    
    def _get_info_latErrMode(self, reward = 0, f_success=False, f_invalid=False):
        return {"current reward": reward, "is_success": f_success, "is_invalid": f_invalid}
    
    def _get_reward_latErrMode(self):
        """Returns the reward for the current state."""
        # get trailer x2, y2, theta2, phi
        trailer_posn = self._ttwr_states[3:5]
        theta2 = self._ttwr_states[5]
        phi = self._ttwr_states[6]
        # get target x2, y2, theta2, phi
        # self._target_state = np.array([-20, -20, 0, 0])
        target_posn = self._target_state[0:2]
        theta_target = self._target_state[2]

        self.dist_to_goal = eucDistCompt(trailer_posn, target_posn)

        f_parked = self._is_success()
        f_invalid = self._is_invalid()
        
        if f_parked:
            r = 100
        elif f_invalid:
            r = -100
        else:
            action_array = np.array([self._velo, self._action])
            # cost from action, avoid large steering
            action_cost = ( action_array.T ).dot( self._action_weight ).dot( action_array )
            # cost from deviation, avoid large deviation
            diviation_cost = ( ( self.deviate_err.T ).dot( self._err_states_weight ) ).dot( self.deviate_err )

            J = diviation_cost + action_cost

            # slight cost for large phi, and positive reward to keep moving
            r = -J + (-0.1) * self._ttwr_states[6] + 1

        return r, f_parked, f_invalid
    
    def _is_success(self):
        """Returns True if the episode is over."""
        # get trailer x2, y2, theta2, phi
        trailer_posn = self._ttwr_states[3:5]
        theta2 = self._ttwr_states[5]
        phi = self._ttwr_states[6]
        # get target x2, y2, theta2, phi
        # self._target_state = np.array([-20, -20, 0, 0])
        target_posn = self._target_state[0:2]
        theta_target = self._target_state[2]

        dist_to_goal = eucDistCompt(trailer_posn, target_posn)
        f_arrival = bool(dist_to_goal <= 0.5)
        align_with_goal = bool(abs(wrapToPi(theta2 - theta_target)) <= 0.2)

        return (f_arrival and align_with_goal)
    
    def _is_invalid(self):
        """Returns True if the current state is invalid."""
        
        # Check if the trailer is outside the map
        if (self._ttwr_states[0] < -self.x2_lim or self._ttwr_states[0] > self.x2_lim or
                self._ttwr_states[1] < -self.y2_lim or self._ttwr_states[1] > self.y2_lim):
            f_out_bound = True
        else:
            f_out_bound = False

        # check if the trailer is jack-knifed
        if ( np.abs( self._ttwr_states[6] ) > self.phi_lim):
            f_jackknife = True
        else:
            f_jackknife = False

        # check if the system is timeout
        if self.step_idx >= self.max_step:
            f_timeout = True
        else:
            f_timeout = False

        # too large distance deviation
        if self.deviate_err[2] >= 5:
            f_dist_deviate = True
        else:
            f_dist_deviate = False

        # too large angle deviation
        if self.deviate_err[1] >= np.pi/6:
            f_angle_deviate = True
        else:
            f_angle_deviate = False

        return (f_out_bound or f_jackknife or f_timeout or f_dist_deviate or f_angle_deviate)
    
    def render(self):
        if self.render_mode == "human": #rgb_array
            self.renderModel.render(self._ttwr_states, self._action, self.path_ref_pnts, self.ref_idx)

    def get_closest_ref_pnt(self, cur_x, cur_y, prev_idx):
        # compute the distance from current positon to previous reference point
        min_index = prev_idx
        min_dist = eucDistCompt(self.path_ref_pnts[prev_idx][0:2], 
                                [cur_x, cur_y])
        f_search_prior_pnts = True

        # reference points prior to current ref point, in case the vehicle is moving opposite direction
        # search_range = 10
        # if prev_idx > search_range:
        #     prev_idx = prev_idx - search_range
        search_end = min(prev_idx + 100, len(self.path_ref_pnts) - 1)

        # look through neighbor ref points to find the closest ref point
        for i in range(prev_idx, search_end):
            cur_dist = eucDistCompt(self.path_ref_pnts[i][0:2],
                                    [cur_x, cur_y])
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_index = i

            # implement look ahead distance
            # https://github.com/ahmedmoawad124/Self-Driving-Vehicle-Control

        min_index = min(min_index + self.look_ahead, len(self.path_ref_pnts) - 1)

        return min_index

    # get the closest reference point and compute the error, input is the current simulation step (index)
    def _get_error_latErrMode(self):
        # get trailer ref point
        cur_trailer_ref_idx = self.get_closest_ref_pnt(self._ttwr_states[3], self._ttwr_states[4],
                                                 self.prev_trailer_ref_idx)
        # debug signal
        self.ref_idx = cur_trailer_ref_idx

        # get host ref point
        cur_host_ref_idx = self.get_closest_ref_pnt(self._ttwr_states[0], self._ttwr_states[1],
                                                    self.prev_host_ref_idx)


        theta2_err = wrapToPi( (self.path_ref_pnts[cur_trailer_ref_idx][2] - self._ttwr_states[5]).item() )
        theta1_err = wrapToPi( (self.path_ref_pnts[cur_host_ref_idx][2] - self._ttwr_states[2]).item() )

        x2_err = (self.path_ref_pnts[cur_trailer_ref_idx][0] - self._ttwr_states[3]).item()
        y2_err = (self.path_ref_pnts[cur_trailer_ref_idx][1] - self._ttwr_states[4]).item()

        transform_mat = np.array([[np.cos(theta2_err), np.sin(theta2_err), 0], 
                                [-np.sin(theta2_err), np.cos(theta2_err), 0],
                                [0                  , 0,                  1]])
        trailer_dist_err = transform_mat.dot(np.array([x2_err, y2_err, theta2_err]).T)

        self.prev_trailer_ref_idx = cur_trailer_ref_idx
        self.prev_host_ref_idx = cur_host_ref_idx

        # only care the lateral error for now
        return np.concatenate( (np.array([theta1_err, theta2_err]), trailer_dist_err[0:2]) , axis = 0)