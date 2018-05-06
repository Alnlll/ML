import numpy as np
import cv2
from arsenal.common.RGB_table import RGB_TABLE
from keras import backend as K
import colorsys
import imghdr
import random
from PIL import Image, ImageDraw, ImageFont

draw_detection_box_default = {
    "rec_info" : {
        "n_h" : 0,
        "n_w" : 0,
        "origin" : [0,0],
        "color" : RGB_TABLE['Green'],
        "thickness": 1,
    },
    "text_info" : {
        "text" : 'default',
        "coordinate" : [0,0],
        "font_type" : cv2.FONT_HERSHEY_SIMPLEX,
        "font_size" : 0.4,
        "font_pixel": (10, 7), # (height, width)
        "color" : RGB_TABLE['Cyan'],
        "thickness" : 1,
        "line_type" : cv2.LINE_AA,
    }
}
def draw_detection_box(image, box_info = draw_detection_box_default):
    '''
    Input:
        image -- cv2 object, represent for the image to draw
        rec_info -- python built-in type dict, height, width, coordinate of left-up corner
                    "n_h" : height
                    "n_w" : width
                    "origin" : (start_h, start_w)
                    "color" : (R, G, B) or scalar for grayscale
                    "thickness" : thickness
        text_info -- python built-in type dict, string, coordinate of bottom-left corner, font type, font scale, color, thickness
                     "text" : string to put
                     "coordinate" : bottom-left corner (start_h, start_w)
                     "font_type"
                     "font_scale"
                     "color" : (R, G, B) or scalar for grayscale
                     "thickness" : thickness
                     "line_type"
    '''

    # Get info
    try:
        rec_info = box_info['rec_info']
        text_info = box_info['text_info']
    except Exception as e:
        raise KeyError("draw_detection_box -- box_info[{}]".format(e))
        #return None

    # Draw box
    try:
        n_h, n_w = rec_info['n_h'], rec_info['n_w']
        origin = tuple(rec_info['origin'][::-1]) # (h, w) to (w, h) for OpenCV
        color = rec_info['color'][::-1] # OpenCV need GBR
        thickness = rec_info['thickness']
        end = (n_w+origin[0], n_h+origin[1])
    except Exception as e:
        raise KeyError("draw_detection_box -- rec_info[{}]".format(e))
        #return None

    cv2.rectangle(image, origin, end, color, thickness)

    # Put text
    try:

        text = text_info['text']

        coor_w, coor_h = origin[0], origin[1] - 5
        if 0 > coor_h: coor_h = origin[1] + 5
        coordinate = (coor_w, origin[1]-5)

        font_type = text_info['font_type']
        font_size = text_info['font_size']
        font_pixel = text_info['font_pixel']
        color = text_info['color'][::-1] # OpenCV need GBR
        thickness = text_info['thickness']
        line_type = text_info['line_type']

    except Exception as e:
        raise KeyError("draw_detection_box -- text_info[{}]".format(e))
        #return None
    #Draw background of text
    cv2.rectangle(image,
                  (coordinate[0], coordinate[1]-font_pixel[0]),
                  (coordinate[0]+font_pixel[1]*len(text), coordinate[1]+2),
                  [255-c for c in color],
                  -1)

    cv2.putText(image, text, coordinate, font_type, font_size, color, thickness, line_type)

    return image

def get_class_names(file_path):

    try:
        f = open(file_path)
        names = f.readlines()
    except Exception as e:
        print("get_class_names -- {}".format(e))
        return None

    return [name.strip() for name in names if name]

def get_anchors(file_path):
    try:
        f = open(file_path)
        line = f.readline()
    except Exception as e:
        print("get_anchors -- {}".format(e))
        return None

    coordinates = line.strip().split(', ')
    coordinates = [float(coor) for coor in coordinates]
    anchors = np.array([coordinates])
    anchors = anchors.reshape(len(coordinates)//2, 2)

    return anchors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def get_image(image_path, image_shape=(720, 1280)):

    image = cv2.imread(image_path)
    imr = cv2.resize(image, image_shape, interpolation=cv2.INTER_CUBIC)

    print(type(imr))

    imr = imr / 255
    imr = np.expand_dims(imr, 0)

    return imr

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
