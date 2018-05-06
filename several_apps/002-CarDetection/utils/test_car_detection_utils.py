
import os,sys
if not ("%s/../.." % os.getcwd()) in sys.path:
    sys.path.append("%s/../../.." % os.getcwd())

import cv2
from car_detection_utils import *
from arsenal.common.RGB_table import RGB_TABLE

def test_draw_detection_box():
    image = cv2.imread('./test.jpg')

    box_info = draw_detection_box_default
    box_info['rec_info']['n_h'] = 400
    box_info['rec_info']['n_w'] = 900
    box_info['rec_info']['origin'] = [100,100]

    box_info['text_info']['text'] = 'Car: 0.92'

    iamge = draw_detection_box(image, box_info=box_info)

    cv2.imshow('Image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_get_image():
    image = get_image('./test.jpg')

    print(image.shape)

options = {
    'draw_box': test_draw_detection_box,
    'get_image': test_get_image,
}

if __name__ == "__main__":
    func = options['get_image']
    func()
