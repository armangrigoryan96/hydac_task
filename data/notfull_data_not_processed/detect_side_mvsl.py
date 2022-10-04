#%%
from glob import glob
from typing import Any
from tensorflow.keras import models
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__)))

#%%
#anomaly_folder = glob('W:/SiegertSensorsAnomalyDetection/Extracted data/Ring_contacts/test/mvsl/*.jpg') #folder with images
anomaly_folder = glob(os.path.join(ROOT_DIR, 'data/*.jpg'))
inference_model_contact_ring_ae = models.load_model(os.path.join(ROOT_DIR, 'models', 'ae_contact_ring_rgb.h5'), compile=False)

def find_side_mvsl_contours(input_black_background, kernel, kernel2):
    "function contains a set of hard coded rules to detect the contours which indicate there is a rotated silicone"
    input_black_background_step1 = cv2.morphologyEx(input_black_background, cv2.MORPH_CLOSE, kernel)
    input_black_background_step2 = cv2.morphologyEx(input_black_background_step1, cv2.MORPH_OPEN, kernel2)

    contours_found, _ = cv2.findContours(input_black_background_step2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_passed = []

    count_contours = len(contours_found)
    areas = [cv2.contourArea(cnt) for cnt in contours_found]
        
    if len(areas)>0 :
        max_contour = max(areas)
        for contour, contour_area in zip(contours_found, areas):
                #contour_area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(contour_area)/hull_area
                #print('Area is %s and solidity is %s' %(contour_area, solidity))
            c = max(contour, key=cv2.contourArea)
            if ( (67<c[0][0]<90 or 410<c[0][0]<440) and 240<c[0][1]<380 and (((320.0 < contour_area < 1030.0) and solidity>0.74) or ((250.0 < contour_area < 320.0) and solidity>0.8) or
                (max_contour>13000.0 and solidity>0.685 and 200.0 < contour_area < 500.0)) ):
                    
                contours_passed.append(contour)
    return contours_found, contours_passed

#%%
i=0
for path in anomaly_folder:
    i+=1
    image = cv2.imread(path) #load input image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb_resized = cv2.resize(image_rgb, (512,512))
    image_rgb_float32 = np.float32(image_rgb_resized)/255
    input_batched_array = np.expand_dims(image_rgb_float32, 0)
    reconstructed_image_float32 = inference_model_contact_ring_ae(input_batched_array) #apply autoencoder model
    reconstructed_image_float32 = np.squeeze(reconstructed_image_float32)
    reconstructed_image_uint8 = np.uint8(reconstructed_image_float32*255)

    #extract black background from the reconstructed image
    reconstructed_rgb_black_bg = cv2.inRange(reconstructed_image_uint8,(25, 25, 35), (75, 75, 96)) 

    kernel_broken = np.ones((5,5),np.uint8)
    kernel_broken2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    #extract contours associated with rotated silicone
    contours_found_recon, contours_passed_recon = find_side_mvsl_contours(reconstructed_rgb_black_bg, kernel_broken, kernel_broken2)

    if len(contours_passed_recon)>0:
   
        reconstructed_image_copy_uint8 = reconstructed_image_uint8.copy()
        cv2.drawContours(reconstructed_image_copy_uint8, contours_found_recon, -1, (255,0,0), 3)
        cv2.drawContours(reconstructed_image_copy_uint8, contours_passed_recon, -1, (0,255,0), 3)

        plt.rcParams['figure.figsize'] = (9,11)
        plt.subplot(1,2,1), plt.imshow(image_rgb_float32)
        plt.subplot(1,2,2), plt.imshow(reconstructed_image_copy_uint8)
        plt.show()

#%%
