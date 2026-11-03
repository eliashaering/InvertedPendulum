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
    // Flush in-memory SRAM log buffer to USB host
    static void flush_logs_to_usb() {
        if (log_stream_buffer == nullptr) return;

        char read_buf[128];
        size_t read_bytes;
        
        while ((read_bytes = xStreamBufferReceive(log_stream_buffer, read_buf, sizeof(read_buf), 0)) > 0) {
            uart_write_bytes(UART_NUM_0, read_buf, read_bytes);
        }
        const char* end_str = "END_LOGS\n";
        uart_write_bytes(UART_NUM_0, end_str, std::strlen(end_str));
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
        
        RxPacket rx_packet;
        const float SCALE = 1000.0f;
        while (true){
            int size = uart_read_bytes(UART_NUM_0, &rx_packet, sizeof(RxPacket), portMAX_DELAY) ;
            if (size == sizeof(RxPacket)) {
                if(rx_packet.header == UART_Header_LOG){
                    flush_logs_to_usb();
                } 
                else if (rx_packet.header == UART_Header_META) { // Send Metadata to Sim 
                    MetaPacket meta_packet;
                    meta_packet.header = UART_Header_META;     
                    meta_packet.flash_part = get_app_partition_size();   
                    meta_packet.sram_total = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);   
                    meta_packet.sram_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL); 
                    meta_packet.min_heap   = esp_get_minimum_free_heap_size();
                    meta_packet.stack_c0 = uxTaskGetStackHighWaterMark(NULL);   
                    meta_packet.stack_c1 = inference_task_handle ? uxTaskGetStackHighWaterMark(inference_task_handle) : 0; 
                    
                    uart_write_bytes(UART_NUM_0, &meta_packet, sizeof(MetaPacket));
                }
                else if (rx_packet.header == UART_Header_OBS){ // Input from Simulation: Observation data
                    Observation obs = {rx_packet.obs[0] / SCALE, rx_packet.obs[1] / SCALE, rx_packet.obs[2] / SCALE, rx_packet.obs[3] / SCALE};
            
                    // Send Observation to Inference 
                    xQueueSend(queues->obs_queue, &obs, portMAX_DELAY);
                    
                    float action;
                    // Receive Action from Inference and send packet to Sim
                    if (xQueueReceive(queues->action_queue, &action, portMAX_DELAY)) {
                        ActPacket act_packet;
                        act_packet.header = UART_Header_ACT;
                        act_packet.action = static_cast<int16_t>(action * SCALE);
                        uart_write_bytes(UART_NUM_0, &act_packet, sizeof(ActPacket));
                     }
                }
                else {
                // Clear hardware buffer on framing error
                uart_flush_input(UART_NUM_0);
                }
            }
        }
    }
};