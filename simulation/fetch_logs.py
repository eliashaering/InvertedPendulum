import serial
import time
import struct

# Configure your specific USB port here
SERIAL_PORT = '/dev/cu.usbserial-3120' # Update with your actual port
BAUD_RATE = 115200

# UART Protocol Constants
UART_Header_LOG = 0x02

def fetch_esp32_logs():
    try:
        # Open serial connection with a 1-second timeout
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as esp32:
            print("Connected to ESP32. Requesting logs...")
            
            # Clear input buffer to remove any old data
            esp32.reset_input_buffer()
            
            # Send the 9-byte binary trigger command (<B4h)
            req_logs = struct.pack('<B4h', UART_Header_LOG, 0, 0, 0, 0)
            esp32.write(req_logs)
            
            print("\n--- ESP32 IN-MEMORY LOGS ---")
            
            # Read text response line by line until the end marker is found
            while True:
                line = esp32.readline().decode('utf-8', errors='replace')
                
                # Stop if timeout occurs (empty line received)
                if not line:
                    print("[Timeout] Log stream interrupted or empty.")
                    break
                    
                # Stop reading when the exact end signal arrives
                if "END_LOGS" in line:
                    break
                    
                # Print the log line directly (end="" prevents double spacing)
                print(line, end="")
                
            print("----------------------------\n")
                
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")

if __name__ == "__main__":
    fetch_esp32_logs()
