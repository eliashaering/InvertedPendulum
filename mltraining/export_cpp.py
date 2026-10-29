import numpy as np
from stable_baselines3 import SAC
import os

# Load trained reinforcement learning model and extract the actor policy network
model = SAC.load("./models/mujoco_cartpole_sac")
actor = model.policy.actor

# Flatten model weights and biases for C++ export
w1 = actor.latent_pi[0].weight.detach().numpy().flatten()
b1 = actor.latent_pi[0].bias.detach().numpy().flatten()
w2 = actor.latent_pi[2].weight.detach().numpy().flatten()
b2 = actor.latent_pi[2].bias.detach().numpy().flatten()
w_mu = actor.mu.weight.detach().numpy().flatten()
b_mu = actor.mu.bias.detach().numpy().flatten()

high = model.action_space.high[0]

# 1. Pruning and 8-bit integer quantization
def quantize_weights(arr, threshold=1e-3):
    # Zero out weights below threshold (pruning) to increase sparsity
    arr[np.abs(arr) < threshold] = 0.0
    max_val = np.max(np.abs(arr))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    # Map float weights to int8 range [-127, 127]
    q_arr = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    sparsity = np.sum(q_arr == 0) / len(q_arr) * 100
    return q_arr, scale, sparsity

q_w1, scale_w1, sp_w1 = quantize_weights(w1)
q_w2, scale_w2, sp_w2 = quantize_weights(w2)
q_w_mu, scale_w_mu, sp_w_mu = quantize_weights(w_mu)

print(f"Sparsity (saved computations): W1: {sp_w1:.1f}%, W2: {sp_w2:.1f}%, W_mu: {sp_w_mu:.1f}%")

# Format numpy arrays as raw C int8 arrays (PROGMEM omitted for ESP-IDF compatibility)
def to_c_array_int8(arr, name):
    vals = ", ".join([str(v) for v in arr])
    return f"const int8_t {name}[{len(arr)}] = {{{vals}}};\n"

# Format bias arrays as C float arrays
def to_c_array_float(arr, name):
    vals = ", ".join([f"{v:.6f}f" for v in arr])
    return f"const float {name}[{len(arr)}] = {{{vals}}};\n"

# 2. Generate optimized C++ header using pointer arithmetic and pre-scaling
cpp_code = f"""#pragma once
#include <math.h>
#include <stdint.h>
// PROGMEM removed for native ESP-IDF compatibility

const float SCALE_W1 = {scale_w1:.8f}f;
const float SCALE_W2 = {scale_w2:.8f}f;
const float SCALE_W_MU = {scale_w_mu:.8f}f;

{to_c_array_int8(q_w1, 'W1')}
{to_c_array_int8(q_w2, 'W2')}
{to_c_array_int8(q_w_mu, 'W_mu')}

{to_c_array_float(b1, 'b1')}
{to_c_array_float(b2, 'b2')}
{to_c_array_float(b_mu, 'b_mu')}

// Fast deterministic inference routine optimized for real-time ESP32 execution
inline float predict_action(const float obs[4]) {{
    float h1[256];
    const int8_t* w1_ptr = W1;

    // First hidden layer: Process inputs with pre-scaled int8 weights and ReLU activation
    for (int i = 0; i < 256; i++) {{
        float sum = b1[i];
        sum += (*w1_ptr++) * SCALE_W1 * obs[0];
        sum += (*w1_ptr++) * SCALE_W1 * obs[1];
        sum += (*w1_ptr++) * SCALE_W1 * obs[2];
        sum += (*w1_ptr++) * SCALE_W1 * obs[3];
        h1[i] = sum > 0.0f ? sum : 0.0f; 
    }}

    float h2[256];
    const int8_t* w2_ptr = W2;
    float h1_scaled[256];
    
    // Pre-scale hidden layer outputs to optimize matrix multiplication loops
    for(int i=0; i<256; i++) h1_scaled[i] = h1[i] * SCALE_W2;

    // Second hidden layer: Fully-connected dense layer with ReLU activation
    for (int i = 0; i < 256; i++) {{
        float sum = b2[i];
        const float* h1_s_ptr = h1_scaled;
        
        for (int j = 0; j < 256; j++) {{
            sum += (float)(*w2_ptr++) * (*h1_s_ptr++);
        }}
        h2[i] = sum > 0.0f ? sum : 0.0f;
    }}

    // Output layer (mean): Compute final linear combination
    float mu = b_mu[0];
    const int8_t* w_mu_ptr = W_mu;
    const float* h2_ptr = h2;
    
    for (int j = 0; j < 256; j++) {{
        mu += (*w_mu_ptr++) * (SCALE_W_MU * (*h2_ptr++));
    }}

    // Apply hyperbolic tangent activation scaled by action space limits
    return tanhf(mu) * {high}f;
}}
"""

output_dir = "../esp32_firmware/src"
os.makedirs(output_dir, exist_ok=True)
with open(f"{output_dir}/network.h", "w") as f:
    f.write(cpp_code)