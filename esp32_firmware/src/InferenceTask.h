#pragma once
#include "Shared.h"
#include "network.h"
#include "esp_timer.h"

class InferenceTask {
public:
    // Launch the inference task pinned exclusively to Core 1 with a dedicated stack
    static void start(SystemQueues* queues) {
        static const char* TAG_Inference = "Inference";
        xTaskCreatePinnedToCore(task_loop, "NN", 16384, queues, 5, nullptr, 1);
    }

private:
    // Main background loop dedicated to real-time neural network execution
    static void task_loop(void* arg) {
        auto queues = static_cast<SystemQueues*>(arg);
        Observation obs;
        
        while (true) {
            // Block until a new observation vector arrives from Core 0
            if (xQueueReceive(queues->obs_queue, &obs, portMAX_DELAY) == pdTRUE) {
                
                // Profile the precise inference execution time using high-res timers
                int64_t t_start = esp_timer_get_time();
                float action = predict_action(obs.data());
                int64_t t_end = esp_timer_get_time();
                
                // Send the computed action back to the communication pipeline
                xQueueSend(queues->action_queue, &action, portMAX_DELAY);
                
                // Log runtime latency and resulting control action
                ESP_LOGD("INFER", "Latency: %lld us | Action: %.3f", (t_end - t_start), action);
            }
        }
    }
};