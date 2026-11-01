#pragma once
#include <array>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

// Define a fixed-size observation vector matching the MuJoCo CartPole state space dimensions
using Observation = std::array<float, 4>;

extern TaskHandle_t inference_task_handle;

// Container structure holding thread-safe FreeRTOS queues for inter-core communication
struct SystemQueues {
    QueueHandle_t obs_queue;    // Queue transmitting sensor observations from Core 0 to Core 1
    QueueHandle_t action_queue; // Queue transmitting computed neural network actions back to Core 0
};