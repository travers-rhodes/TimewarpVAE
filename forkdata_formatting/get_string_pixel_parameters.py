import cv2
import numpy as np

# take in an image which contains a white circular plate and darker "food"
# and return the center/radius of the plate and the
# center/ellipse parameters of the (darker) food
# return a tuple specifying facts about the plate and about the ellipse fitting the thread
# note that the angle is returned here in degrees, not radians, for the ellipse
# return looks like: ((plate_center, plate_radius),(mean_inds,axislens, th / np.pi * 180))
# all in units of pixels
# Given the camera position for photos taken on 2022-08-08
# The mean_inds is a tuple of (x,y) with 
#      x in the -x direction of vicon/world
#      and y in the y direction of vicon/world
# The axis lens are the full lengths (diameters) along major/minor axes.
# The rotation angle (in degrees) gives the rotation FROM IMAGE X TOWARD Y
#     (that is, the _clockwise_ rotation of the first (major) axis).
def process_image(filename, plate_erosion=0.8, zscore_detection=2):
    src = cv2.imread(filename)
    #https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                           param1=100, param2=30,
                           minRadius=40, maxRadius=80)
    #https://stackoverflow.com/questions/36911877/cropping-circle-from-image-using-opencv-python
    height,width = src.shape[:2]
    mask = np.zeros((height,width), np.uint8)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]: # for now, just take first circle
            plate_center = (i[0], i[1])
            plate_radius = np.uint16(i[2]*plate_erosion)
        circle_img = cv2.circle(mask,plate_center,plate_radius,(1,1,1),thickness=-1)
    else:
        raise Exception("no circle identified")
    # model most of image as gaussian.
    # significant outliers are the thread
    image_grayscale = gray[circle_img==1]
    mean = np.mean(image_grayscale)
    std = np.std(image_grayscale)
    thread_indices = (gray < mean - zscore_detection * std) * (circle_img==1)
    thread_locations = np.zeros((height,width), np.uint8)
    thread_locations[thread_indices] = 1
    # https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    dilation_size=3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                           (dilation_size, dilation_size))
    dilation_dst = cv2.dilate(thread_locations, element)
    inds = np.array(np.where(thread_locations==1)).T
    # image x y is different ordering from numpy x y (I guess)
    inds = np.hstack((inds[:,1:],inds[:,:1]))
    mean_inds = np.mean(inds,axis=0)
    centered_inds = inds - mean_inds


    u,s,vt = np.linalg.svd(np.cov(centered_inds.T))
    axislens = np.sqrt(s) * 3
    th = np.arccos(u[0,0])
    # unclear what angle ellipse _wants_ defined, but this empirically works to pick the right quadrant
    if u[0,1] < 0:
      th *= -1

    ellipse = (mean_inds,axislens, th / np.pi * 180)

    # return a tuple specifying facts about the plate and about the ellipse fitting the thread
    # note that the angle is returned here in degrees, not radians, for the ellipse
    return ((plate_center, plate_radius),ellipse)
