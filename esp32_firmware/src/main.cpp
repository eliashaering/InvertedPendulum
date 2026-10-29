#include "Shared.h"
#include "CommsTask.h"
#include "InferenceTask.h"

extern "C" void app_main(void) {
    // Define logging tag for the startup sequence
    static const char* TAG_Init = "Initialization";

    // Allocate queues statically to ensure their lifetime across the entire execution
    static SystemQueues queues;
    
    // Initialize non-blocking RAM ring buffer before any logs are generated
    init_in_memory_logging();

    // Create FreeRTOS queues with a depth of 1 for immediate real-time synchronization
    queues.obs_queue = xQueueCreate(1, sizeof(Observation));
    queues.action_queue = xQueueCreate(1, sizeof(float));

    // Launch core-pinned background tasks and inject queue references
    CommsTask::start(&queues);
    InferenceTask::start(&queues);

    // Confirm successful system boot
    ESP_LOGI(TAG_Init, "Initialization complete");
}