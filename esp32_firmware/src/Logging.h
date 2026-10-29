#include <stdio.h>
#include <stdarg.h>
#include "freertos/FreeRTOS.h"
#include "freertos/stream_buffer.h"
#include "esp_log.h"

// Global handle for the log stream buffer
StreamBufferHandle_t log_stream_buffer = nullptr;

// Custom hook to intercept ESP_LOG output
int in_memory_log_vprintf(const char *fmt, va_list args) {
    if (log_stream_buffer == nullptr) return 0;

    // Static stack buffer prevents fatal heap fragmentation during runtime
    char temp_buf[256]; 
    
    // Format the final log string
    int len = vsnprintf(temp_buf, sizeof(temp_buf), fmt, args);
    
    if (len > 0) {
        // Prevent buffer overflow
        if (len >= sizeof(temp_buf)) len = sizeof(temp_buf) - 1;
        
        // Write to RAM. Timeout is 0: drop logs silently if full to prevent blocking.
        // Blocking here (e.g., portMAX_DELAY) would crash the real-time control loop.
        xStreamBufferSend(log_stream_buffer, temp_buf, len, 0);
    }
    
    return len;
}

void init_in_memory_logging() {
    // Allocate 4KB RAM buffer with a 1-byte trigger level
    log_stream_buffer = xStreamBufferCreate(4096, 1);
    
    if (log_stream_buffer != nullptr) {
        // Redirect all ESP_LOG output from hardware UART to RAM
        esp_log_set_vprintf(in_memory_log_vprintf);
    }
}