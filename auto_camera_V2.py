"""
The code below is a template for the auto_camera.py file. You will need to
finish the capture() function to take a picture at a given RPY angle. Make
sure you have completed the sensor_calc.py file before you begin this one.
"""

#import libraries
from adafruit_lsm6ds.lsm6dsox import LSM6DSOX as LSM6DS
from adafruit_lis3mdl import LIS3MDL
import time
import os
import board
import busio
import sys
sys.path.append('/usr/lib/python3/dist-packages')

from picamera2 import Picamera2
import numpy as np
import sys
from git import Repo
from sensor_calc_V2 import *
from crater_detection import *
from path_finding import *

#imu and camera initialization
i2c = busio.I2C(board.SCL, board.SDA)
accel_gyro = LSM6DS(i2c)
mag = LIS3MDL(i2c)
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

#VARIABLES
THRESHOLD = 3               # Any desired value from the accelerometer
REPO_PATH = "/home/cubesat/Documents/CubeSat/"      # Your github repo path: ex. /home/pi/FlatSatChallenge
FOLDER_PATH = "Images"      # Your image folder path in your GitHub repo: ex. /Images

def git_push():
    """
    This function is complete. Stages, commits, and pushes new images to your GitHub repo.
    """
    try:
        print("pushing to git")
        repo = Repo(REPO_PATH)
        origin = repo.remote('origin')
        print('added remote')
        origin.pull()
        print('pulled changes')
        repo.git.add(REPO_PATH) # + FOLDER_PATH
        repo.index.commit('New Photo')
        print('made the commit')
        origin.push()
        print('pushed changes')
    except Exception as e:
        print('Couldn\'t upload to git: ', e)

def img_gen(name):
    """
    This function is complete. Generates a new image name.

    Parameters:
        name (str): your name ex. MasonM
    """
    t = time.strftime("_%y_%m_%d_%H%M%S")
    imgname = (f'{REPO_PATH}/{FOLDER_PATH}/{name}{t}.jpg')
    return imgname

def take_photo():
    """
    Takes a photo when the FlatSat is shaken.
    """

    accelx, accely, accelz = accel_gyro.acceleration

    time.sleep(1)
    name = "CarissaP"
    img_name = img_gen(name)
    picam2.start_and_capture_file(img_name)

    time.sleep(1)
    return img_name

#Code to take a picture at a given offset angle
def capture(dir ='roll', target_angle = 30):
    #Calibration lines should remain commented out until you implement calibration
    #offset_mag = calibrate_mag()
    #offset_gyro =calibrate_gyro()
    offset_mag = [0, 0, 0]
    offset_gyro = [0, 0, 0]
    initial_angle = set_initial(offset_mag)
    prev_angle = initial_angle
    roll_angle = initial_angle[0]
    print("Begin moving camera.")
    while True:
        accelX, accelY, accelZ = accel_gyro.acceleration #m/s^2
        magX, magY, magZ = mag.magnetic #gauss
        # print(accelX, accelY, accelZ)
	    #Calibrate magnetometer readings
        magX = magX - offset_mag[0]
        magY = magY - offset_mag[1]
        magZ = magZ - offset_mag[2]
        gyroX, gyroY, gyroZ = accel_gyro.gyro #rad/s
        #Convert to degrees and calibrate
        gyroX = gyroX *180/np.pi - offset_gyro[0]
        gyroY = gyroY *180/np.pi - offset_gyro[1]
        gyroZ = gyroZ *180/np.pi - offset_gyro[2]
        #print(gyroX, gyroY, gyroZ);
        #print(magX, magY, magZ);
        
        delT = 1
        roll_angle += gyroX * delT
        prev_angle = [gyroX, gyroY, gyroZ]
        print(roll_angle);
        #print(f"AccelX:{accelX:6.2f} accelY:{accelY:6.2f} AccelZ:{accelZ:6.2f} | "
        #  f"GyroX:{gyroX:6.2f} GyroY:{gyroY:6.2f} GyroZ:{gyroZ:6.2f}")
        
        #if dir == 'roll' and abs(roll_angle - target_angle) < 3:
        img_name = take_photo();
        
        img_bgr = cv2.imread(img_name)
        if img_bgr is None:
            raise ValueError("Could not load the image.")
        
        # Turn the image into grayscale
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # image in grayscale and bgr
        ml_img_ORIG = cv2.resize(img_bgr, (512, 512)) # bgr
        ml_img = cv2.resize(gray, (512, 512)) # gray
        
        [crater_path, crater_contour] = crater_detection(img_name, ml_img_ORIG, ml_img)
        path_finding(img_name, crater_path, crater_contour);
        
        git_push()
        
        time.sleep(20);

if __name__ == '__main__':
    capture(*sys.argv[1:])
    
