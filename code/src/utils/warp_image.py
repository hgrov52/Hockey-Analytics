import cv2
import numpy as np



TRANSFORM_X = 200
TRANSFORM_Y = 200
TRANSFORM_X_SHAPE = 20
TRANSFORM_Y_SHAPE = 5
MAX_SHAPE = 400

# lines must be in the form (theta, rho) or [theta, rho]
# returns two vars: x,y
def find_intersection_point(line1,line2):
    A=np.array([[np.cos(line1[0]),np.sin(line1[0])],
                [np.cos(line2[0]),np.sin(line2[0])]])
    B=np.array([[line1[1]],
                [line2[1]]])
    x0, y0 = np.linalg.solve(A, B)
    return int(np.round(x0)), int(np.round(y0))

def find_point_along_line(x0,y0,theta,constant = 100, constant_x = None, constant_y = None):
    a = np.cos(theta)
    b = np.sin(theta)

    if(constant_x!=None and constant_y != None):
        x1 = int(x0 - constant_x*(-b))
        y1 = int(y0 - constant_y*(a))
        return x1,y1


    x1 = int(x0 - constant*(-b))
    y1 = int(y0 - constant*(a))
    return x1,y1

# line in the form (theta, rho) or [theta, rho]
def draw_line(im,line):
    theta,rho = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1500*(-b))
    y1 = int(y0 + 1500*(a))
    x2 = int(x0 - 1500*(-b))
    y2 = int(y0 - 1500*(a))
    cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)

def rotate_points_cw(x00,y00,x01,y01,x11,y11,x10,y10):
    return x01,y01,x11,y11,x10,y10,x00,y00

def rotate_points_ccw(x00,y00,x01,y01,x11,y11,x10,y10):
    return x10,y10,x00,y00,x01,y01,x11,y11

def generate_points(im, warp_lines, draw=False):
    top = left = bottom = right = None
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
        top = min(horiz_lines,key=lambda x:x[1])
        # side of the ice
        if(len(steep_pos_slope)>0):
            left = max(steep_pos_slope,key=lambda x:x[0])
            steep_pos_slope.remove(left)
            for x in steep_pos_slope:
                extra_vertical_line.append(x)
            for x in steep_neg_slope:
                extra_vertical_line.append(x)
        # right side of the ice
        elif(len(steep_neg_slope)>0):
            left = max(steep_neg_slope,key=lambda x:x[0])
            steep_neg_slope.remove(left)
            for x in steep_neg_slope:
                extra_vertical_line.append(x)
            for x in steep_pos_slope:
                extra_vertical_line.append(x)
        
    if(len(extra_vertical_line)>0):
        right = extra_vertical_line[0]

    print("\ntop | lft | btm | rht")
    print(top!=None,left!=None,bottom!=None,right!=None)

    if(left == None or top == None):
        return


    NEW_TRANSFORM = 840

    x00,y00 = find_intersection_point(top,left)
    #if(draw):
        #cv2.circle(im,(x00,y00),5,(0,255,0),-1)
        
    

    if(right != None):
        x01,y01 = find_point_along_line(x00,y00,left[0],-TRANSFORM_Y)
        x10,y10 = find_intersection_point(top,right)
        x11,y11 = find_point_along_line(x10,y10,right[0],-TRANSFORM_Y)
    else:
        x01,y01 = find_point_along_line(x00,y00,left[0],-TRANSFORM_Y)
        x10,y10 = find_point_along_line(x00,y00,top[0],TRANSFORM_X)
        x11,y11 = find_point_along_line(x10,y10,left[0],-TRANSFORM_Y)
    #if(draw):
        # cv2.circle(im,(x01,y01),5,(0,255,0),-1)
        # cv2.circle(im,(x10,y10),5,(0,255,0),-1)
        # cv2.circle(im,(x11,y11),5,(0,255,0),-1)

    return np.float32(((x00,y00), (x01,y01), (x11,y11), (x10,y10)))


def get_ratio(blue_lines,yellow_line):
    # find radio between real life distance and image distance
    return
    

def warp_image(im,warp_lines, draw=False, params = None):
    if(params is not None):
        TRANSFORM_X = params[0]
        TRANSFORM_Y = params[1]
        TRANSFORM_X_SHAPE = params[2]
        TRANSFORM_Y_SHAPE = params[3]
        MAX_SHAPE = params[4]

    
    if(draw):
        for line in warp_lines:
            draw_line(im,line)

    

    X_SHAPE = MAX_SHAPE/TRANSFORM_X_SHAPE
    Y_SHAPE = MAX_SHAPE/TRANSFORM_Y_SHAPE
    

    pts1 = generate_points(im,warp_lines,draw)
    
    if(pts1 is None):
        return

    constant = 10
    scale = 10,17
    X_SHIFT = pts1[0][0]/2
    Y_SHIFT = pts1[0][1]/2

    dest_pts = np.float32([[X_SHIFT, Y_SHIFT], [X_SHIFT, scale[1]*constant+Y_SHIFT], [scale[0]*constant+X_SHIFT, scale[1]*constant+Y_SHIFT], [scale[0]*constant+X_SHIFT, Y_SHIFT]])

    
    pts2 = np.float32(((MAX_SHAPE-X_SHAPE, MAX_SHAPE-Y_SHAPE), 
                       (MAX_SHAPE-X_SHAPE, MAX_SHAPE+Y_SHAPE),
                       (MAX_SHAPE+X_SHAPE, MAX_SHAPE+Y_SHAPE), 
                       (MAX_SHAPE+X_SHAPE, MAX_SHAPE-Y_SHAPE)))
        
    #pts2 = np.float32(((0, 0),(0, len(im[0])),(len(im), len(im[0])), (len(im), 0)))

    M = cv2.getPerspectiveTransform(pts1,dest_pts)
    warp = cv2.warpPerspective(im,M,(1000,1000))
    cv2.imshow('final',warp)
    