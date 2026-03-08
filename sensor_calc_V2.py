"""
The code you will write for this module should calculate
roll, pitch, and yaw (RPY) and calibrate your measurements
for better accuracy. Your functions are split into two activities.
The first is basic RPY from the accelerometer and magnetometer. The
second is RPY using the gyroscope. Finally, write the calibration functions.
Run plot.py to test your functions, this is important because auto_camera.py 
relies on your sensor functions here.
"""

#import libraries
import time
import numpy as np
import time
import os
import board
import busio
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS
from adafruit_lis3mdl import LIS3MDL

#imu initialization
i2c = busio.I2C(board.SCL, board.SDA)
accel_gyro = LSM6DS(i2c)
mag = LIS3MDL(i2c)


#Activity 1: RPY based on accelerometer and magnetometer
def roll_am(accelX,accelY,accelZ):
    roll = np.atan(accelY/np.sqrt(accelY*accelY + accelZ*accelZ));
    return roll

def pitch_am(accelX,accelY,accelZ):
    pitch = np.atan(accelX/np.sqrt(accelY*accelY + accelZ*accelZ));
    return pitch

def yaw_am(accelX,accelY,accelZ,magX,magY,magZ):
    pitch = pitch_am(accelX,accelY,accelZ);
    roll = roll_am(accelX, accelY, accelZ)
    mag_x = magX*np.cos(pitch) + magY*np.sin(roll)*np.sin(pitch) + magZ*np.cos(roll)*np.sin(pitch);
    mag_y = magY*np.cos(roll) - magZ*np.sin(roll);
    # (180/np.pi)*np.arctan2(-magY, magX)
    yaw = np.atan(-mag_y/mag_x);
    return yaw

#Activity 2: RPY based on gyroscope
def roll_gy(prev_angle, delT, gyro):
    roll = prev_angle + gyro * delT;
    return roll
def pitch_gy(prev_angle, delT, gyro):
    pitch = prev_angle + gyro * delT;
    return pitch
def yaw_gy(prev_angle, delT, gyro):
    yaw = prev_angle + gyro * delT;
    return yaw

#Activity 3: Sensor calibration
def calibrate_mag():
    #TODO: Set up lists, time, etc
    print("Preparing to calibrate magnetometer. Please wave around.")
    gx = []
    gy = []
    gz = []
    time.sleep(3)
    print("Calibrating...")
    num_samples = 500

    mag_x = []
    mag_y = []
    mag_z = []

    for _ in range(num_samples):
        mx, my, mz = mag.magnetic  # gauss
        mag_x.append(mx)
        mag_y.append(my)
        mag_z.append(mz)
        time.sleep(0.01)

    x_offset = (max(mag_x) + min(mag_x)) / 2
    y_offset = (max(mag_y) + min(mag_y)) / 2
    z_offset = (max(mag_z) + min(mag_z)) / 2
    
    print("Calibration complete.")
    return [x_offset,y_offset,z_offset]

def calibrate_gyro():
    #TODO
    print("Preparing to calibrate gyroscope. Put down the board and do not touch it.")
    num_samples = 500;
    gx = []
    gy = []
    gz = []
    time.sleep(3)
    print("Calibrating...")
    #TODO: Calculate calibration constants
    for _ in range(num_samples):
        gyroX, gyroY, gyroZ = accel_gyro.gyro  # rad/s
        gx.append(gyroX * (180/np.pi))
        gy.append(gyroY * (180/np.pi))
        gz.append(gyroZ * (180/np.pi))
        time.sleep(0.01)
    
    biasX = np.mean(gx)
    biasY = np.mean(gy)
    biasZ = np.mean(gz)
    
    print("Calibration complete.")
    return [biasX,biasY,biasZ]
    return [0, 0, 0]

def set_initial(mag_offset = [0,0,0]):
    """
    This function is complete. Finds initial RPY values.

    Parameters:
        mag_offset (list): magnetometer calibration offsets
    """
    #Sets the initial position for plotting and gyro calculations.
    print("Preparing to set initial angle. Please hold the IMU still.")
    time.sleep(3)
    print("Setting angle...")
    accelX, accelY, accelZ = accel_gyro.acceleration #m/s^2
    magX, magY, magZ = mag.magnetic #gauss
    #Calibrate magnetometer readings. Defaults to zero until you
    #write the code
    magX = magX - mag_offset[0]
    magY = magY - mag_offset[1]
    magZ = magZ - mag_offset[2]
    roll = roll_am(accelX, accelY,accelZ)
    pitch = pitch_am(accelX,accelY,accelZ)
    yaw = yaw_am(accelX,accelY,accelZ,magX,magY,magZ)
    print("Initial angle set.")
    print(roll, pitch, yaw)
    return [roll,pitch,yaw]
