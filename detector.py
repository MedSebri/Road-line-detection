import matplotlib.pylab as plt
import cv2
import numpy as np


def draw_lines(img, lines):
    img=np.copy(img)
    blank_img=np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    
    img=cv2.addWeighted(img, 0.8, blank_img, 1, 0.0)
    return img


def interest_region(img, vertices):
    mask=np.zeros_like(img)
    #channel_count=img.shape[2]
    match_mask_color=255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image=cv2.bitwise_and(img, mask)
    return masked_image

#img=cv2.imread('ld.jpg')
#img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def process(img):
    print(img.shape)
    height=img.shape[0]
    width=img.shape[1]

    region=[
        (0, height),
        (width/2, height/2),
        (width, height) 
    ]

    gray_img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    canny_img=cv2.Canny(gray_img, 100, 120)  
    masked_img=interest_region(canny_img, np.array([region], np.int32))

    lines=cv2.HoughLinesP(masked_img, rho=2, theta=np.pi/60, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=100)

    img_with_lines=draw_lines(img, lines)

    return img_with_lines


cap=cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    ret, frame=cap.read()
    frame=process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  

#plt.imshow(img_with_lines)
#plt.show()
