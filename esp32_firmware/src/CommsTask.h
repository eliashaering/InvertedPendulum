#pragma once
#include <cstdio>
#include <cstring>
#include "driver/uart.h"
#include "Shared.h"
#include "Logging.h"

class CommsTask {
public:
    static void start(SystemQueues* queues) {
        xTaskCreatePinnedToCore(task_loop, "COMMS", 8192, queues, 5, nullptr, 0);
    }

private:
    // Static method required to be callable from the FreeRTOS task loop
    static void flush_logs_to_usb() {
        if (log_stream_buffer == nullptr) return;

        char read_buf[128];
        size_t read_bytes;
        
        // Drain the stream buffer until empty (returns 0)
        while ((read_bytes = xStreamBufferReceive(log_stream_buffer, read_buf, sizeof(read_buf), 0)) > 0) {
            // Output raw bytes directly to UART, bypassing the ESP-IDF logging subsystem
            printf("%.*s", read_bytes, read_buf);
        }
    }

    static void task_loop(void* arg) {
        auto queues = static_cast<SystemQueues*>(arg);
        uart_driver_install(UART_NUM_0, 256, 0, 0, nullptr, 0);
        
        uint8_t data;
        char rx_buf[128]; 
        int idx = 0;
        
        while (true) {
            if (uart_read_bytes(UART_NUM_0, &data, 1, portMAX_DELAY) > 0) {
                if (data == '\n') {
                    // Null-terminate the received payload
                    rx_buf[idx] = '\0'; 
                    
                    // Intercept diagnostic log requests
                    if (strncmp(rx_buf, "GET_LOGS", 8) == 0) {
                        flush_logs_to_usb();
                    } 
                    // Process standard observation vectors for inference
                    else {
                        Observation obs;
                        if (std::sscanf(rx_buf, "%f,%f,%f,%f", &obs[0], &obs[1], &obs[2], &obs[3]) == 4) {
                            xQueueSend(queues->obs_queue, &obs, portMAX_DELAY);
                            
                            float action;
                            if (xQueueReceive(queues->action_queue, &action, portMAX_DELAY)) {
                                char tx_buf[32];
                                int len = std::snprintf(tx_buf, sizeof(tx_buf), "ACT:%.4f\n", action);
                                uart_write_bytes(UART_NUM_0, tx_buf, len);
                            }
                        }else{
                            ESP_LOGE("COMMS", "Parse Error. Ungültiges Paket: %s", rx_buf);
                        }
                    }
                    // Reset buffer index for the next incoming transmission
                    idx = 0; 
                } else if (data != '\r' && idx < 127) {
                    rx_buf[idx++] = data;
                }
            }
        }
    }
};