import time
import serial
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC

from mltraining.train_sac import MuJoCoCartPoleEnv 

# ==========================================
# CONFIGURATION
# ==========================================
USE_ESP32 = True  # True: ESP32 hardware inference | False: Local Python inference
SERIAL_PORT = '/dev/cu.usbserial-3120' # Adjust to match your system's port
# ==========================================

env = MuJoCoCartPoleEnv()

if USE_ESP32:
    esp32 = serial.Serial(SERIAL_PORT, 115200, timeout=1)
    time.sleep(2) # Wait for ESP32 to reboot and initialize
else:
    model = SAC.load("mltraining/models/mujoco_cartpole_sac")

obs, _ = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        # 1. Determine action based on selected backend
        if USE_ESP32:

            msg = f"{obs[0]:.4f},{obs[1]:.4f},{obs[2]:.4f},{obs[3]:.4f}\n"
            esp32.write(msg.encode('utf-8'))

            print(f"Request: {msg}")
            
            esp_response = esp32.readline().decode('utf-8').strip()
            
            action = np.array([0.0]) # Default fallback action
    
            if esp_response:
                # Flush incoming stream until the "ACT:" prefix is encountered
                while not esp_response.startswith("ACT:"):
                    if esp_response.startswith("LOG:"):
                        print(f"ESP32-Log: {esp_response[4:].strip()}")
                    
                    # Fetch the next line from the UART buffer
                    esp_response = esp32.readline().decode('utf-8').strip()
                    
                    if not esp_response:
                        continue # Handle serial read timeout
                
                # Parse the action payload once the prefix is found
                if esp_response and esp_response.startswith("ACT:"):
                    try:
                        action_value = float(esp_response.replace("ACT:", ""))
                        print(f"ESP32-Action: {action_value}")
                        action = np.array([action_value])
                    except ValueError:
                        pass
        else:
            print(f"SIM: Executed action: {action}")
            action, _ = model.predict(obs, deterministic=True)
        
        # 2. Execute physics step
        obs, reward, terminated, truncated, _ = env.step(action)
        viewer.sync()
        
        if terminated or truncated:
            obs, _ = env.reset()
            
        # 3. Enforce real-time execution step rate
        time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)