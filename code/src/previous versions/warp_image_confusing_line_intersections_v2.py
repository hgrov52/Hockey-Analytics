import cv2
import numpy as np

"""
    if(params == None):
        TRANSFORM_X = 200
        TRANSFORM_Y = 600
        TRANSFORM_X_SHAPE = 20
        TRANSFORM_Y_SHAPE = 5
        MAX_SHAPE = 400
    else:
        TRANSFORM_X = params[0]
        TRANSFORM_Y = params[1]
        TRANSFORM_X_SHAPE = params[2]
        TRANSFORM_Y_SHAPE = params[3]
        MAX_SHAPE = params[4]


"""

TRANSFORM_X = 200
TRANSFORM_Y = 600
TRANSFORM_X_SHAPE = 20
TRANSFORM_Y_SHAPE = 5
MAX_SHAPE = 400

def generate_points(im, warp_lines, draw=False):
    """

    mark lines as left, right, top, bottom

    """
    A = []
    B = []
    x00 = y00 = x10 = y10 = x01 = y01 = left = None
    steep_pos_slope = []
    steep_neg_slope = []
    horiz_lines = []
    extra_vertical_line = []
    for theta,rho in warp_lines:
        # steep pos slope
        if(.91<theta and theta<1.39):
            steep_pos_slope.append((theta,rho))
        # steep neg slope
        if(1.85<theta and theta<2.88):
            steep_neg_slope.append((theta,rho))
        # horizontal slopes
        if(1.45<theta and theta<1.72):
            horiz_lines.append((theta,rho))
    if(len(horiz_lines)>0):
        # side of the ice
        if(len(steep_pos_slope)>0):
            left = max(steep_pos_slope,key=lambda x:x[0])
            top = min(horiz_lines,key=lambda x:x[1])
            A.append([np.cos(left[0]),np.sin(left[0])])
            B.append([left[1]])
            A.append([np.cos(top[0]),np.sin(top[0])])
            B.append([top[1]])
            steep_pos_slope.remove(left)
            for x in steep_pos_slope:
                extra_vertical_line.append(x)
            for x in steep_neg_slope:
                extra_vertical_line.append(x)
        # right side of the ice
        elif(len(steep_neg_slope)>0):
            left = max(steep_neg_slope,key=lambda x:x[0])
            top = min(horiz_lines,key=lambda x:x[1])
            A.append([np.cos(left[0]),np.sin(left[0])])
            B.append([left[1]])
            A.append([np.cos(top[0]),np.sin(top[0])])
            B.append([top[1]])
            steep_neg_slope.remove(left)
            for x in steep_neg_slope:
                extra_vertical_line.append(x)
            for x in steep_pos_slope:
                extra_vertical_line.append(x)


    if(len(A)>0 and len(B)>0):
        # solve for intersection point
        A = np.array(A)
        B = np.array(B)
        x00, y00 = np.linalg.solve(A, B)
        x00, y00 = int(np.round(x00)), int(np.round(y00))
        
        if(draw):
            cv2.circle(im, (x00, y00), 5, (0, 255, 0), -1)
    A = []
    B = []
    # if 3 lines
    if(len(extra_vertical_line)>0):
        top = min(horiz_lines,key=lambda x:x[1])
        A.append([np.cos(extra_vertical_line[0][0]),np.sin(extra_vertical_line[0][0])])
        B.append([extra_vertical_line[0][1]])
        A.append([np.cos(top[0]),np.sin(top[0])])
        B.append([top[1]])
        if(len(A)>0 and len(B)>0):
            # solve for intersection point
            A = np.array(A)
            B = np.array(B)
            x10, y10 = np.linalg.solve(A, B)
            x10, y10 = int(np.round(x10)), int(np.round(y10))
        
            if(draw):
                cv2.circle(im, (x10, y10), 5, (0, 255, 0), -1)

    
    """
    A.append([np.cos(theta),np.sin(theta)])
    B.append([rho])
        

    # solve for intersection point
    A = np.array(A)
    B = np.array(B)
    x00, y00 = np.linalg.solve(A, B)
    x00, y00 = int(np.round(x00)), int(np.round(y00))
    
    
    """

def warp_image(im,warp_lines, draw=False):



    if(draw):
        for theta,rho in warp_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1500*(-b))
            y1 = int(y0 + 1500*(a))
            x2 = int(x0 - 1500*(-b))
            y2 = int(y0 - 1500*(a))
            cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)

    

    X_SHAPE = MAX_SHAPE/TRANSFORM_X_SHAPE
    Y_SHAPE = MAX_SHAPE/TRANSFORM_Y_SHAPE
    

    pts1 = generate_points(im,warp_lines,draw)
    """
    if(draw):
        cv2.circle(im,pts1[0],5,(0,255,0),-1)
        cv2.circle(im,pts1[1],5,(0,255,0),-1)
        cv2.circle(im,pts1[2],5,(0,255,0),-1)
        cv2.circle(im,pts1[3],5,(0,255,0),-1)
    
    pts2 = np.float32(((MAX_SHAPE-X_SHAPE, MAX_SHAPE-Y_SHAPE), 
                       (MAX_SHAPE-X_SHAPE, MAX_SHAPE+Y_SHAPE),
                       (MAX_SHAPE+X_SHAPE, MAX_SHAPE+Y_SHAPE), 
                       (MAX_SHAPE+X_SHAPE, MAX_SHAPE-Y_SHAPE)))
    M = cv2.getPerspectiveTransform(pts1,pts2)
    warp = cv2.warpPerspective(im,M,(1000,1000))
    #cv2.imshow('final',warp)
    """