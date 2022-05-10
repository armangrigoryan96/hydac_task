import matplotlib.pyplot as plt
import numpy as np
import cv2
import math



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
        dilate = cv2.dilate(thresh, kernel, iterations=40)
        return newImage, dilate

    def get_angle(self, x1,y1,x2,y2):
        myradians = math.atan2(y1-y2, x1-x2)
        mydegrees = math.degrees(myradians)
        return mydegrees

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    # find contours of the detected features in the image and separate them all into a bounding box with 4 points: A,B,C,D. See explaination below
    def get_coord(self, newImage, dilate):
        contours, hierarchy = cv2.findContours(dilate,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # image = cv2.drawContours(newImage, contours, -1, (0, 255, 0), 10)

        horizontal_coordinates =[]
        vertical_coordinates =[]

        for contour in contours:
            min_horiz, max_horiz = np.argmin(contour.squeeze()[:,0], axis=0), np.argmax(contour.squeeze()[:,0], axis=0)
            min_vert, max_vert = np.argmin(contour.squeeze()[:,1], axis=0), np.argmax(contour.squeeze()[:,1], axis=0)

            horizontal_coordinates.append([contour[min_horiz], contour[max_horiz]])
            vertical_coordinates.append([contour[min_vert], contour[max_vert]])

        #removing the biggest circle contour
        horizontal_coordinates = np.array(horizontal_coordinates[1:])
        vertical_coordinates = np.array(vertical_coordinates[1:])

        # Getting the bounding box endpoints of the shaded area, A-left, B-right, C-top, D-bottom
        A_point = horizontal_coordinates[np.argmin(horizontal_coordinates.squeeze()[:,0][:,0])].squeeze()[0]
        B_point = horizontal_coordinates[np.argmax(horizontal_coordinates.squeeze()[:,0][:,0])].squeeze()[0]
        C_point = vertical_coordinates[np.argmin(vertical_coordinates.squeeze()[:,0][:,0])].squeeze()[0]
        D_point = vertical_coordinates[np.argmax(vertical_coordinates.squeeze()[:,0][:,0])].squeeze()[0]

        x1_hor, y1_hor = A_point
        x2_hor, y2_hor = B_point
        x1_vert, y1_vert = C_point
        x2_vert, y2_vert = D_point

        bbox_w = abs(x2_hor-x1_hor)
        bbox_h = abs(y2_vert-y1_vert)

        x_radius, y_radius, _ = newImage.shape
        x_radius, y_radius = x_radius//2, y_radius//2
        w,h = newImage.shape[0],newImage.shape[1]

        return bbox_w,bbox_h,w,h,\
           x1_hor, y1_hor, x2_hor,\
           y2_hor, x1_vert,y1_vert,\
           x2_vert,y2_vert, x_radius, y_radius

    # applying a geometrical solution to comparisons to understand weather the features found in the image dilate are located at the top/bottom/left/right
    def find_rotation_angle(self, newImage, dilate):
        angle=0
        bbox_w,bbox_h,w,h, x1_hor, y1_hor, x2_hor, y2_hor, x1_vert,y1_vert, x2_vert,y2_vert, x_radius, y_radius = self.get_coord(newImage, dilate )

        if bbox_w-bbox_h>=0:
            if y_radius-max(y1_vert, y2_vert)>=0: # then the boxes lie at the top
                regulator = 5
                angle=180+self.get_angle(x1_hor,y1_hor,x2_hor,y2_hor) + regulator
                print('top', angle)
            elif y_radius-max(y1_vert, y2_vert)<0: # then the boxes are at the bottom
                regulator=-5
                angle=self.get_angle(x1_hor,y1_hor,x2_hor,y2_hor) + regulator
                print('bottom', angle)

        elif bbox_w-bbox_h<0:
            if x_radius-max(x1_hor, x2_hor)>=0: #then the box is situated at the left side

                bbox_w,bbox_h,w,h, x1_hor, y1_hor, x2_hor, y2_hor, x1_vert,y1_vert, x2_vert,y2_vert,x_radius, y_radius = self.get_coord(self.rotate_image(newImage, 90), self.rotate_image(dilate, 90))
                regulator = -5

                angle = 90+self.get_angle(x1_hor,y1_hor,x2_hor,y2_hor) + regulator
                print('left', angle)
            elif x_radius-max(x1_hor, x2_hor)<0: #then the box is situated at the right side
                regulator=5
                bbox_w,bbox_h,w,h, x1_hor, y1_hor, x2_hor, y2_hor, x1_vert,y1_vert, x2_vert,y2_vert,x_radius, y_radius = self.get_coord(self.rotate_image(newImage, 90), self.rotate_image(dilate, 90))

                angle = -90 + self.get_angle(x1_hor,y1_hor,x2_hor,y2_hor) + regulator

                print('right', angle)
        return angle

    def save_aligned_image(self, newImage, angle, file_name):
        # plt.imshow(self.rotate_image(newImage, angle))
        # plt.show()
        cv2.imwrite(f'{self.output_path}/{file_name.split("/")[-1]}', self.rotate_image(newImage, angle))

    # the final to go over all methods in class ImageAligner
    def align(self, file_name):
        image, dilate = self.get_image_dilate(file_name)
        angle = self.find_rotation_angle(image, dilate)
        self.save_aligned_image(image, angle, f'{self.input_path}/{file_name}')

