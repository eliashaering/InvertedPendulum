#pragma once
#include <array>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

// Define a fixed-size observation vector matching the MuJoCo CartPole state space dimensions
using Observation = std::array<float, 4>;

// Handle to access the inference task from the comms side
extern TaskHandle_t inference_task_handle;

// Header Definitions for UART-Comm
const uint8_t UART_Header_OBS = 0x00;   //Observation (SIM -> ESP)
const uint8_t UART_Header_ACT = 0x01;   //Observation (ESP -> SIM)
const uint8_t UART_Header_LOG = 0x02;   //Observation (ESP -> SIM)
const uint8_t UART_Header_META = 0x03;  //Observation (ESP -> SIM)

// Packet from SIM -> ESP
struct __attribute__((packed)) RxPacket
{
    uint8_t header;
    int16_t obs[4];
};

// Packet from ESP -> SIM
struct __attribute__((packed)) ActPacket
{
    uint8_t header;
    int16_t action;
};

struct __attribute__((packed)) MetaPacket {
    uint8_t header;       
    uint32_t flash_part; 
    uint32_t sram_total;
    uint32_t sram_free; 
    uint32_t min_heap; 
    uint32_t stack_c0;   
    uint32_t stack_c1;     
};


// Container structure holding thread-safe FreeRTOS queues for inter-core communication
struct SystemQueues {
    QueueHandle_t obs_queue;    // Queue transmitting sensor observations from Core 0 to Core 1
    QueueHandle_t action_queue; // Queue transmitting computed neural network actions back to Core 0
};