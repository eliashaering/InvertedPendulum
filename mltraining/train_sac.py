import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from stable_baselines3 import SAC
import os

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
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Initialize cart at the center of the track
        self.data.qpos[0] = 0.0 
        # Initialize pole with minimal random jitter
        self.data.qpos[1] = 0.0 + np.random.uniform(-0.05, 0.05) 

        self.current_step = 0
        mujoco.mj_forward(self.model, self.data)
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
        
        # Compute reward based on pole height/angle (0.0 at bottom, 1.0 upright)
        reward = (1.0 - np.cos(pole_angle)) / 2.0
        
        # Apply penalties only when the pole is in the upper half
        if np.cos(pole_angle) < 0:
            reward -= 0.01 * (pole_vel**2)           # Angular velocity damping penalty
            reward -= 0.001 * (self.data.ctrl[0]**2) # Control effort penalty for smooth actuation
            
        # Minor penalty to keep the cart centered on the rail
        reward -= 0.01 * (cart_pos**2)

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
    env = MuJoCoCartPoleEnv()
    # Initialize Soft Actor-Critic (SAC) reinforcement learning agent
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")
    # Train the neural network policy over multiple steps
    model.learn(total_timesteps=600000, progress_bar = True) 
    model.save("mujoco_cartpole_sac")