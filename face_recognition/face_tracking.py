import cv2
import numpy as np
from scipy.spatial.distance import pdist
from skimage.measure import compare_ssim
import dlib
import sys
import os
import math
#from runnable.distance_metric import *

person1 = cv2.imread('../pictures/face_1.png')
person2 = cv2.imread('../pictures/face_2.png')
person3 = cv2.imread('../pictures/face_3.png')
person4 = cv2.imread('../pictures/face_4.png')
person5 = cv2.imread('../pictures/face_5.png')
person6 = cv2.imread('../pictures/face_6.png')

def display_image(img):
    cv2.namedWindow("Image") 
    cv2.imshow("Image", img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()

def rotate_image_reverse(coordinate, M):
    A = M[:, 0:2]
    B = M[:, 2]
    coordinate -= B
    A_inv = np.linalg.inv(A)
    return np.dot(A_inv, np.transpose(coordinate))

def rotate_image(img, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2) 

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated = cv2.warpAffine(img, M, (nW, nH))
    return rotated, M

def resize_image(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

def mahalanobis_distance(vec0, vec1):
    combo = np.array([vec0, vec1])
    return pdist(combo, 'mahalanobis')

def edge_detector(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    lap_gray = cv2.cvtColor(lap, cv2.COLOR_RGB2GRAY)
    display_image(lap_gray)

def face_detector_opencv(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # lap = cv2.Laplacian(img,cv2.CV_64F)#拉普拉斯边缘检测 
    # lap = np.uint8(np.absolute(lap))##对lap去绝对值
    # lap_gray = cv2.cvtColor(lap, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.08, 2)
    return faces

def face_detector_dlib(img):
    detector = dlib.get_frontal_face_detector()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dets = detector(img, 1)
    return dets

def face_selector(img):
    checkin = False
    angles = 30
    while True:
        angles -= 30
        processed_img, M = rotate_image(img, angles)
        # faces = face_detector_dlib(processed_img)
        # for index, face in enumerate(faces):
        #     checkin = True
        #     left = face.left()
        #     top = face.top()
        #     right = face.right()
        #     bottom = face.bottom()
        #     cv2.rectangle(processed_img, (left, top), (right, bottom), (0, 255, 0), 3)
        faces = face_detector_opencv(processed_img)
        for (x, y, w, h) in faces:
            checkin = True
            # cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0))
        if checkin == True or angles <= -360:
            break

    if checkin == True:
        return faces, angles, processed_img, M
    else:
        return None

def find_peopleID(clientID, zed1, zed0, id):
    """
    给出当前拍摄到的图像，和所给的id进行匹配，检测是否是当前的人脸
    :param zed1: 无人机相机左目视觉
    :param zed0: 无人机相机右目视觉
    :type  ndarrays
    :param id: 目标任务的id
    :type  int

    :return: 人脸的位置，相当于世界坐标系
    :rtype: [x, y, z] with respect to the world
            return None for no faces in the picture
    """
    faces1, angles1, processed_img1, M1 = face_selector(zed1)
    faces0, angles0, processed_img0, M0 = face_selector(zed0)
    people = []
    max_sim = -1
    max_id = -1
    if faces1 is None or faces0 is None:
        return None
    else:
        for i in range(0, min(len(faces1), len(faces0))):
            x, y, w, h = faces1[i]
            x0, y0, w0, h0 = faces0[i]
            print(faces0)
            print(faces1)
            face1 = processed_img1[y:y+h, x:x+w, :]
            #face0 = processed_img0[y:y+h, x:x+w, :]
            people.append(resize_image(person1, w, h))
            people.append(resize_image(person2, w, h))
            people.append(resize_image(person3, w, h))
            people.append(resize_image(person4, w, h))
            people.append(resize_image(person5, w, h))
            people.append(resize_image(person6, w, h))
            for j in range(0, len(people)):
                sim = compare_ssim(face1, people[j], multichannel=True)
                if sim > max_sim:
                    max_sim = sim
                    max_id = j + 1
            print(max_id)
            if max_id == id:
                coord1 = rotate_image_reverse([x+w/2, y+h/2], M1)
                coord0 = rotate_image_reverse([x0+w0/2, y0+h0/2], M0)
                return reprojectionTo3D(clientID, coord1, coord0)
    return None

if __name__ == '__main__':
    zed1 = cv2.imread('../pictures/people2.jpeg')
    find_peopleID(0, zed1, zed1, 2)
    display_image(zed1)

