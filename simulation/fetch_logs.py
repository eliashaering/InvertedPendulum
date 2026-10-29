import serial
import time

# Configure your specific USB port here
SERIAL_PORT = '/dev/cu.usbserial-3120' # Update with your actual port
BAUD_RATE = 115200

def fetch_esp32_logs():
    try:
        # Open serial connection with a 1-second timeout
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as esp32:
            print("Connected to ESP32. Requesting logs...")
            
            # Send the trigger command defined in C++
            esp32.write(b"GET_LOGS\n")
            
            # Give the ESP32 a few milliseconds to process and flush the buffer
            time.sleep(0.1)
            
            # Read all available bytes from the UART buffer
            raw_data = esp32.read_all()
            
            if raw_data:
                # Decode bytes to string and print
                logs = raw_data.decode('utf-8', errors='replace')
                print("\n--- ESP32 IN-MEMORY LOGS ---")
                print(logs)
                print("----------------------------\n")
            else:
                print("Buffer empty. No logs received.")
                
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")

if __name__ == "__main__":
    fetch_esp32_logs()