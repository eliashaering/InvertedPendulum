#pragma once
#include <cstdio>
#include <cstring>
#include "driver/uart.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_ota_ops.h"
#include "esp_app_desc.h" 
#include "Shared.h"
#include "Logging.h"

// Handles UART communication and diagnostic requests on Core 0
class CommsTask {
public:
    static void start(SystemQueues* queues) {
        xTaskCreatePinnedToCore(task_loop, "COMMS", 8192, queues, 5, nullptr, 0);
    }

private:
    // Helper to write raw byte arrays directly over UART0
    static void send_response(const char* str, size_t len = 0) {
        if (str == nullptr) return;
        if (len == 0) {
            len = std::strlen(str);
        }
        uart_write_bytes(UART_NUM_0, str, len);
    }

    // Flush in-memory SRAM log buffer to USB host
    static void flush_logs_to_usb() {
        if (log_stream_buffer == nullptr) return;

        char read_buf[128];
        size_t read_bytes;
        
        while ((read_bytes = xStreamBufferReceive(log_stream_buffer, read_buf, sizeof(read_buf), 0)) > 0) {
            send_response(read_buf, read_bytes);
        }
        send_response("END_LOGS\n");
    }

    // Query active partition size in bytes
    static uint32_t get_app_partition_size() {
        const esp_partition_t *running = esp_ota_get_running_partition();
        return running ? running->size : 0;
    }

    // Main UART RX/TX loop
    static void task_loop(void* arg) {
        auto queues = static_cast<SystemQueues*>(arg);
        uart_driver_install(UART_NUM_0, 256, 0, 0, nullptr, 0);
        
        uint8_t data;
        char rx_buf[128]; 
        int idx = 0;
        
        while (true) {
            if (uart_read_bytes(UART_NUM_0, &data, 1, portMAX_DELAY) > 0) {
                if (data == '\n') {
                    rx_buf[idx] = '\0'; 
                    
                    if (strncmp(rx_buf, "GET_LOGS", 8) == 0) {
                        flush_logs_to_usb();
                    } 
                    else if (strncmp(rx_buf, "GET_META", 8) == 0) {
                        uint32_t flash_part = get_app_partition_size();
                        uint32_t sram_total = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
                        uint32_t sram_free  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
                        uint32_t min_heap   = esp_get_minimum_free_heap_size();
                        uint32_t stack_c0   = uxTaskGetStackHighWaterMark(NULL);
                        uint32_t stack_c1   = inference_task_handle ? uxTaskGetStackHighWaterMark(inference_task_handle) : 0;

                        char meta_buf[256];
                        int len = std::snprintf(meta_buf, sizeof(meta_buf),
                            "META:CPU=%d,OPT=%d,FLASH_PART=%lu,SRAM_USED=%lu,HEAP_FREE=%lu,HEAP_MIN=%lu,STK0=%lu,STK1=%lu\n",
                            CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
                            #ifdef __OPTIMIZE__
                            1,
                            #else
                            0,
                            #endif
                            (unsigned long)flash_part,
                            (unsigned long)(sram_total - sram_free),
                            (unsigned long)sram_free,
                            (unsigned long)min_heap,
                            (unsigned long)stack_c0,
                            (unsigned long)stack_c1);

                        send_response(meta_buf, len);
                    }
                    else {
                        Observation obs;
                        if (std::sscanf(rx_buf, "%f,%f,%f,%f", &obs[0], &obs[1], &obs[2], &obs[3]) == 4) {
                            xQueueSend(queues->obs_queue, &obs, portMAX_DELAY);
                            
                            float action;
                            if (xQueueReceive(queues->action_queue, &action, portMAX_DELAY)) {
                                char tx_buf[32];
                                int len = std::snprintf(tx_buf, sizeof(tx_buf), "ACT:%.4f\n", action);
                                send_response(tx_buf, len);
                            }
                        } else {
                            ESP_LOGE("COMMS", "Parse Error: %s", rx_buf);
                        }
                    }
                    idx = 0; 
                } else if (data != '\r' && idx < 127) {
                    rx_buf[idx++] = data;
                }
            }
        }
    }
};