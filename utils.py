import numpy as np
import cv2
import re
from matplotlib import pyplot as plt


def biggest_contour(contours,min_area):
    biggest = None
    max_area = 0
    biggest_n=0
    approx_contour=None
    for n,i in enumerate(contours):
            area = cv2.contourArea(i)
            if area > min_area/10:
                    peri = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.02*peri,True)
                    if area > max_area and len(approx)==4:
                            biggest = approx
                            max_area = area
                            biggest_n=n
                            approx_contour=approx
                            
                                                  
    return biggest_n,approx_contour


def four_point_transform(image, pts):

    pts=pts.reshape(4,2)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped



def transformation(image):
  image=image.copy()  
  height, width, channels = image.shape
  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  image_size=gray.size
  
  gray = cv2.GaussianBlur(gray,(3,3),2)
  threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
  threshold = cv2.fastNlMeansDenoising(threshold, 11, 31, 9)
  
  edges = cv2.Canny(threshold,50,150,apertureSize = 7)
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  simplified_contours = []


  for cnt in contours:
      hull = cv2.convexHull(cnt)
      simplified_contours.append(cv2.approxPolyDP(hull,
                                0.001*cv2.arcLength(hull,True),True))
  simplified_contours = np.array(simplified_contours)
  biggest_n,approx_contour = biggest_contour(simplified_contours,image_size)

  threshold = cv2.drawContours(image, simplified_contours ,biggest_n, (0,255,0), 1)

  dst = 0
  if approx_contour is not None and len(approx_contour)==4:
      approx_contour=np.float32(approx_contour)
      dst= four_point_transform(threshold,approx_contour)
  croppedImage = dst
  
  return croppedImage



def final_image(rotated,value):
  kernel_sharpening = np.array([[0,-1,0], 
                                [-1, 5,-1],
                                [0,-1,0]])
  sharpened = cv2.filter2D(rotated, -1, kernel_sharpening)
  grey = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
  ret1,th = cv2.threshold(grey,value,255,cv2.THRESH_BINARY)
  mw = cv2.adaptiveThreshold(th,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) 
  sharpened = cv2.fastNlMeansDenoising(mw, 11, 31, 9)
  
  return sharpened
