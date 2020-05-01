import cv2
import numpy as np

def print_ground():
    ground = np.matrix([[]])
    return ground


def maping(actual_x, actual_y, actual_z):
    x = focal_l*actual_x/actual_z+Cx
    y = focal_l*actual_y/actual_z+Cy
    return (int(y),int(x))

def final():
    global screen_size, h, Z, focal_l, Cx, Cy, background_width

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    screen_size = (1300,650)
    screen_width, screen_hight = screen_size
    h = 1.6     #assumtion, unit:[m], hight of the camera
    Z = 300     #assumtion, unit:[m], far end distance
    focal_l = 3900
    print("focal_l is: ", focal_l)
    R = 5

    cx = int(screen_width//2.0)
    cy = int(375)
    background_width = 10
    final_img = cv2.imread("Springtime-Golf-Course_AmeriTurf_2019.jpg")
    final_img = cv2.resize(final_img, screen_size)
    for Z in reversed(range(1,350,10)):

        r_x = int(5 * focal_l / Z)
        r_y = int(1.6 * focal_l / Z)

        cv2.circle(final_img,(cx,cy), 10, (255,255,0))
        cv2.circle(final_img, (cx+r_x, cy+r_y),10,(255,255,0))
        cv2.circle(final_img, (cx-r_x,cy+r_y),10,(255, 255, 0))
        cv2.line(final_img, (cx+r_x, cy+r_y), (cx-r_x,cy+r_y), (255, 255, 0), 2)

        cv2.line(final_img, (cx -r_x, cy ), (cx - r_x, cy + r_y), (255, 255, 0), 2)
        cv2.line(final_img, (cx +r_x, cy), (cx + r_x, cy+r_y), (255, 255, 0), 2)
        cv2.imshow(",", final_img)
        cv2.waitKey(100)
    cv2.waitKey(10000)



def main():
    final()

if __name__ == "__main__":
    main()
