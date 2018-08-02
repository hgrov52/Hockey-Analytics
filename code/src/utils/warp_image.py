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

def generate_points(warp_lines, draw=False):

    A = []
    B = []
    for theta,rho in warp_lines:
        #print(theta*180/np.pi,rho)
        A.append([np.cos(theta),np.sin(theta)])
        B.append([rho])
        if(draw):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1500*(-b))
            y1 = int(y0 + 1500*(a))
            x2 = int(x0 - 1500*(-b))
            y2 = int(y0 - 1500*(a))
            cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)

    # solve for intersection point
    A = np.array(A)
    B = np.array(B)
    x00, y00 = np.linalg.solve(A, B)
    x00, y00 = int(np.round(x00)), int(np.round(y00))


    # 1.414 theta threshold found by averaging 
    # the averages between the two measured ranges 
    # of positive sloped thetas
    # 1.3788 - 1.4661
    # 1.3614 - 1.4486
    #        ->        1.42245 & 1.405
    # so if theta is greater, rotate warping box cc once
    if(warp_lines[0][0]>1.414):
        #TRANSFORM_X = 200
        #TRANSFORM_Y = 200
        a_pos = np.cos(warp_lines[0][0])
        b_pos = np.sin(warp_lines[0][0])
        a_neg = np.cos(warp_lines[1][0])
        b_neg = np.sin(warp_lines[1][0])

        x01 = int(x00 + TRANSFORM_X*(-b_pos))
        y01 = int(y00 + TRANSFORM_X*(a_pos))
        x10 = int(x00 - TRANSFORM_Y*(-b_neg))
        y10 = int(y00 - TRANSFORM_Y*(a_neg))

        x11 = int(x01 - TRANSFORM_Y*(-b_neg))
        y11 = int(y01 - TRANSFORM_Y*(a_neg))

        tmp_x00,tmp_y00 = x00,y00
        x00,y00 = x01,y01
        x01,y01 = x11,y11
        x11,y11 = x10,y10
        x10,y10 = tmp_x00,tmp_y00
    else:
        #TRANSFORM_X = 200
        #TRANSFORM_Y = 600
        a_pos = np.cos(warp_lines[0][0])
        b_pos = np.sin(warp_lines[0][0])
        a_neg = np.cos(warp_lines[1][0])
        b_neg = np.sin(warp_lines[1][0])

        x01 = int(x00 + TRANSFORM_Y*(-b_pos))
        y01 = int(y00 + TRANSFORM_Y*(a_pos))
        x10 = int(x00 - TRANSFORM_X*(-b_neg))
        y10 = int(y00 - TRANSFORM_X*(a_neg))
        

        x11 = int(x01 - TRANSFORM_X*(-b_neg))
        y11 = int(y01 - TRANSFORM_X*(a_neg))
    return np.float32(((x00,y00), (x01,y01), (x11,y11), (x10,y10)))


def warp_image(im,warp_lines, draw=False):
    X_SHAPE = MAX_SHAPE/TRANSFORM_X_SHAPE
    Y_SHAPE = MAX_SHAPE/TRANSFORM_Y_SHAPE

    pts1 = generate_points(warp_lines)
    """
    if(draw):
        cv2.circle(im,pts1[0],5,(0,255,0),-1)
        cv2.circle(im,pts1[1],5,(0,255,0),-1)
        cv2.circle(im,pts1[2],5,(0,255,0),-1)
        cv2.circle(im,pts1[3],5,(0,255,0),-1)
    """
    pts2 = np.float32(((MAX_SHAPE-X_SHAPE, MAX_SHAPE-Y_SHAPE), 
                       (MAX_SHAPE-X_SHAPE, MAX_SHAPE+Y_SHAPE),
                       (MAX_SHAPE+X_SHAPE, MAX_SHAPE+Y_SHAPE), 
                       (MAX_SHAPE+X_SHAPE, MAX_SHAPE-Y_SHAPE)))
    M = cv2.getPerspectiveTransform(pts1,pts2)
    warp = cv2.warpPerspective(im,M,(1000,1000))
    cv2.imshow('final',warp)