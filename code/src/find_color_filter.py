import cv2
import numpy as np


def find_color_values(im, operation = 'sub'):
    im_minus_roi = im.copy()
    # Select multiple ROI
    roi = np.zeros(im.shape,dtype=np.uint8)
    roi.fill(255)
    boxes = []
    while(1):
        r = cv2.selectROI(im_minus_roi,False)

        # if no selection, break
        if(r[2] == 0 and r[3] == 0):
            break

        # add box to roi
        for i in range(r[2]):
            for j in range(r[3]):
                roi[j+r[1]][i+r[0]] = im[j+r[1]][i+r[0]]
                im_minus_roi[j+r[1]][i+r[0]] = 255

        
        #cv2.imshow('roi',roi)
        #cv2.imshow('no roi',im_minus_roi)
        boxes.append(r)   

    best = (0,0,0,0)
    
    highest_ratio = -len(im)*len(im[0])

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    no_roi_hsv = cv2.cvtColor(im_minus_roi, cv2.COLOR_BGR2HSV)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    print("finding color values...")
    for r in range(0,255,20):
        
        for g in range(0,255,20):

            for b in range(0,255,20):
                #print("pass 1:",r,g,b, highest_ratio)
                for ra in range(0,60,10):
                    lower_range = np.array([r-ra, g-ra, b-ra], dtype=np.uint8)
                    upper_range = np.array([r+ra, g+ra, b+ra], dtype=np.uint8)
                    
                    roi_mask = cv2.inRange(roi_hsv, lower_range, upper_range)
                    no_roi_mask = cv2.inRange(no_roi_hsv, lower_range, upper_range)
                    
                    nonzero_roi = np.count_nonzero(roi_mask)
                    nonzero_no_roi = np.count_nonzero(no_roi_mask)

                    if(nonzero_roi == 0 and nonzero_no_roi == 0):
                        continue

                    if(operation in ['d','div','division'] and nonzero_no_roi!=0):
                        ratio = nonzero_roi/nonzero_no_roi
                    elif(operation in ['s','sub','subtraction']):
                        ratio = nonzero_roi - nonzero_no_roi
                    

                    if(ratio>highest_ratio):
                        highest_ratio = ratio
                        best = (r,g,b,ra)



    for r in range(best[0]-10,best[0]+10,2):
        if(r<0):
            continue
        
        for g in range(best[1]-10,best[1]+10,2):

            if(g<0):
                continue
            for b in range(best[2]-10,best[2]+10,2):
                if(b<0):
                    continue
                #print("pass 2:",r,g,b, highest_ratio)
                for ra in range(best[3]-5,best[3]+5,2):
                    if(ra<0):
                        continue
                    lower_range = np.array([r-ra, g-ra, b-ra], dtype=np.uint8)
                    upper_range = np.array([r+ra, g+ra, b+ra], dtype=np.uint8)
                    roi_mask = cv2.inRange(roi_hsv, lower_range, upper_range)
                    no_roi_mask = cv2.inRange(no_roi_hsv, lower_range, upper_range)
                    
                    nonzero_roi = np.count_nonzero(roi_mask)
                    nonzero_no_roi = np.count_nonzero(no_roi_mask)

                    if(nonzero_roi == 0 and nonzero_no_roi == 0):
                        continue

                    if(operation in ['d','div','division'] and nonzero_no_roi!=0):
                        ratio = nonzero_roi/nonzero_no_roi
                    elif(operation in ['s','sub','subtraction']):
                        ratio = nonzero_roi - nonzero_no_roi
                    

                    if(ratio>highest_ratio):
                        highest_ratio = ratio
                        best = (r,g,b,ra)


    lower_range = np.array([best[0]-best[3], best[1]-best[3], best[2]-best[3]], dtype=np.uint8)
    upper_range = np.array([best[0]+best[3], best[1]+best[3], best[2]+best[3]], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    print(best)
    cv2.imshow('mask',mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return best