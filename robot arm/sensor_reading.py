import serial
import csv
from datetime import datetime

port = 'COM8'  # Configure to same port as in Arduino IDE
ser = serial.Serial(port, 115200) 

try:
    with open("robot arm/serial_sensor_log.csv", mode='x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Sensor_1", "Sensor_2"])
except FileExistsError:
    pass

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            # Skip empty lines
            if not line:
                continue
                
            sensor_data = line.split(',')
            current_time = datetime.now().strftime("%H:%M:%S")
            row_to_write = [current_time] + sensor_data
            
            # Open and close inside the loop to guarantee data is saved immediately
            with open("robot arm/serial_sensor_log.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row_to_write)
            
            print(f"Logged to CSV: {row_to_write}")
                
except KeyboardInterrupt:
    print("Logging stopped.")
    ser.close()