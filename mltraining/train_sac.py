import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from stable_baselines3 import SAC
import os
import sys
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MuJoCoCartPoleEnv(gym.Env):
    """Custom Gymnasium environment for a MuJoCo-based CartPole simulation."""
    def __init__(self):
        super().__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Resolve absolute path to the MuJoCo XML asset file
        xml_path = os.path.join(current_dir, "../simulation/assets/cartpole.xml")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Action Space: Motor torque applied to the cart (-20.0 to 20.0 N)
        self.action_space = spaces.Box(low=-20.0, high=20.0, shape=(1,), dtype=np.float32)
        
        # State Space: [Cart Position, Cart Velocity, Pole Angle, Pole Velocity]
        high = np.array([1.5, 10.0, np.pi, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # 1. Bypass deterministic seeding: Prevent Stable-Baselines3 (VecEnv) 
        # from locking the environment to a fixed reproducible state.
        super().reset(seed=None)
        
        self.current_step = 0
        
        # 2. Prevent state leakage: Flush MuJoCo's internal kinematic memory 
        # to clear residual forces, accelerations, and actuator states from the previous episode.
        mujoco.mj_resetData(self.model, self.data)
        
        # 3. OS-level Entropy: Break SB3's global random seed lock by forcing 
        # the native random module to source true entropy from the operating system (e.g., /dev/urandom).
        random.seed()  
        
        # 4. Initial State Randomization (Domain Randomization): Inject stochasticity 
        # to prevent the agent from overfitting to a single deterministic starting trajectory.
        random_cart_pos = random.uniform(-0.5, 0.5)
        random_pole_angle = random.uniform(-np.pi, np.pi)
        random_cart_vel = random.uniform(-0.1, 0.1)
        random_pole_vel = random.uniform(-0.1, 0.1)

        # Apply the randomized joint positions (qpos) and velocities (qvel) to the state vector
        self.data.qpos[0] = random_cart_pos
        self.data.qpos[1] = random_pole_angle
        self.data.qvel[0] = random_cart_vel
        self.data.qvel[1] = random_pole_vel

        # 5. Kinematic Forward Pass: Force MuJoCo to synchronize the injected 
        # state variables with its internal physics pipeline (e.g., updating spatial coordinates).
        mujoco.mj_forward(self.model, self.data)
        
        # Debugging hook to verify state injection logic
        #print(f"DEBUG RESET: Target Angle = {random_pole_angle:.2f} rad | MuJoCo State = {self.data.qpos[1]:.2f} rad")

        return self._get_obs(), {}



    def step(self, action):
        # Clip and apply control action to the cart motor actuator
        self.data.ctrl[0] = np.clip(action[0], -20.0, 20.0)
        
        # Advance physics simulation step
        for _ in range(1):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1
        obs = self._get_obs()
        cart_pos, cart_vel, pole_angle, pole_vel = obs

        # Map the angle to a continuous error metric (0.0 when perfectly upright at +/- pi, 2.0 when hanging down at 0)
        angle_error = 1.0 + np.cos(pole_angle) 

        # Apply Squared Exponential (Gaussian RBF) reward shaping.
        # This maps the LQR-style quadratic costs to a strictly bounded [0, 1] reward,
        # providing a smooth gradient that stabilizes the SAC actor network.
        reward = np.exp(-(
            2.0  * (angle_error**2) +          # Primary objective: Heavily penalize angular deviation
            0.1  * (pole_vel**2) +             # Damping penalty: Restrict angular velocity to prevent oscillations
            0.5  * (cart_pos**2) +             # Spatial regularization: Force the cart to stay near the center
            0.01 * (self.data.ctrl[0]**2)      # Control effort penalty: Encourage energy-efficient, smooth motor usage
        ))

        # Episode termination and truncation conditions
        terminated = bool(abs(cart_pos) > 3.0)
        truncated = bool(self.current_step >= 1000)

        if terminated:
            reward -= 100.0
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Extract core state variables from MuJoCo data structures
        return np.array([
            self.data.qpos[0], 
            self.data.qvel[0], 
            self.data.qpos[1], 
            self.data.qvel[1]
        ], dtype=np.float32)

if __name__ == "__main__":
    #env = MuJoCoCartPoleEnv()
    env = make_vec_env(MuJoCoCartPoleEnv, n_envs=4, seed = 42)
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0
    )
    # Initialize Soft Actor-Critic (SAC) reinforcement learning agent
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")
    # Train the neural network policy over multiple steps
    model.learn(total_timesteps=1000000, progress_bar = True) 
    model.save("./models/mujoco_cartpole_sac")
    env.save("./models/vec_normalize.pkl")