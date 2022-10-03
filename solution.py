

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from tqdm import tqdm
from pathlib import Path
import argparse
import os


class ImageAligner:
    '''
    Applying image dilate to merge certain features of the images into meaningful information.
    Then according to the coordinates of the features find the optiomal angle to align the image.
    '''

    def __init__(self, input_path, output_path):
        self.input_path=input_path
        self.output_path=output_path

    # find the dilate of the image
    def get_image_dilate(self, file_name):
        newImage = cv2.imread(f'{self.input_path}/{file_name}')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # dilate = cv2.dilate(thresh, kernel, iterations=40)
        # contours, hierarchy = cv2.findContours(dilate,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        crop_target_1 = thresh[500:850, 170:450]
        crop_target_2 = thresh[500:850, 1530:1810]

        fig=plt.figure()
        # fig.add_subplot(3,1,1)
        # plt.imshow(blur)
        # for contour in contours:
        #    cv2.drawContours(newImage, contour, -1, (0, 255, 0), 6)
        fig.add_subplot(3,2,1)

        plt.imshow(crop_target_1)

        # fig=plt.figure()
        #
        # fig.add_subplot(2,1,1)
        # plt.imshow(blur)
        #
        fig.add_subplot(3,2,2)
        plt.imshow(crop_target_2)


        fig.add_subplot(3,1,2)
        plt.imshow(thresh)
        #
        fig.add_subplot(3,1,3)
        plt.imshow(newImage)

        # plt.show()
        fig.savefig(f'{self.output_path}/{file_name}')


        print()
        # return newImage, dilate, hierarchy

    def save_aligned_image(self, newImage, angle, file_name):
        plt.imshow(self.rotate_image(newImage, angle))
        # plt.show()
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(f'{self.output_path}/{file_name.split("/")[-1]}', self.rotate_image(newImage, angle))

    # the final to go over all methods in class ImageAligner
    def align(self, file_name):
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.get_image_dilate(file_name)
        # angle = self.find_rotation_angle(image, dilate)
        # self.save_aligned_image(image, angle, f'{self.input_path}/{file_name}')


parser = argparse.ArgumentParser(description='Aligning the images according to the text')
parser.add_argument('--input_path', type=str,  help='Input path', default = 'test')
parser.add_argument('--output_path', type=str,  help='Output path', default = 'crops_test')

args = parser.parse_args()

if __name__ == '__main__':
    img_aligner = ImageAligner(input_path=args.input_path, output_path=args.output_path)

    #process all the images in the input path and align them accordingly
    for img_filename in tqdm(os.listdir(args.input_path)):
        if img_filename.split('.')[-1] in ['jpg', 'jpeg', 'png']:
            img_aligner.align(img_filename)
