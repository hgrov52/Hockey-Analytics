import numpy as np
imageA = np.zeros((2,6,3))
imageB = np.zeros((10,2,3))

# move last axis to firsts index
#imageA = np.moveaxis(imageA,-1,0)
#imageB = np.moveaxis(imageB,-1,0)

imageA[:,:,0]=1
imageA[:,:,1]=2
imageA[:,:,2]=3
#print('imageA:\n',imageA,'\n')

imageB[:,:,0]=4
imageB[:,:,1]=5
imageB[:,:,2]=6
#print('imB:\n',imageB,'\n')

avgA = (300,300)
avgB = (299,299)

shift_x = avgA[0]-avgB[0]
shift_y = avgA[1]-avgB[1]
right,shift_x = (abs(shift_x),0) if shift_x<0 else (0,shift_x) 
down,shift_y = (abs(shift_y),0) if shift_y<0 else (0,shift_y) 
result = np.zeros((max(imageA.shape[0]+shift_y,imageB.shape[0]+down),max(imageA.shape[1]+shift_x,imageB.shape[1]+right),3))
# result = np.zeros((max(imageA.shape[0],imageB.shape[0])+max(shift_y,down),max(imageA.shape[1],imageB.shape[1])+max(shift_x,right)))
result[:,:]=-np.inf
result[shift_y:(imageA.shape[0]+shift_y),shift_x:(imageA.shape[1]+shift_x),:]=\
np.where(result[shift_y:(imageA.shape[0]+shift_y),shift_x:(imageA.shape[1]+shift_x),:]==-np.inf,\
imageA,result[shift_y:(imageA.shape[0]+shift_y),shift_x:(imageA.shape[1]+shift_x),:])
#print("result before\n",result)
result[down:(imageB.shape[0]+down),right:(imageB.shape[1]+right),:]= \
np.where(result[down:(imageB.shape[0]+down),right:(imageB.shape[1]+right),:]==-np.inf, \
imageB,result[down:(imageB.shape[0]+down),right:(imageB.shape[1]+right),:])
#print('result after\n',result)

print(np.moveaxis(result,-1,0))
