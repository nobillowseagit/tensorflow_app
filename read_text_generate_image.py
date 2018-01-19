import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
image_path_train = "data/image/train/"
if not os.path.exists(image_path_train):
    os.makedirs(image_path_train)
image_path_test = "data/image/test/"
if not os.path.exists(image_path_test):
    os.makedirs(image_path_test)
DEFAULT_FONTS = "data/DroidSansMono.ttf"
WIDHT = 28
HEIGHT = 28
def get_content_from_file(label_name):
    content = open("data/" + label_name, "r", encoding="utf-8")
    code_text = content.read()
    return code_text.split("#")
# 用opencv 转为灰度
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
def generate_image(i, c, dir_path):
    path = dir_path + str(i) + ".jpg"
    print(path)
    color = (0, 0, 0)
    background = (255, 255, 255)
    print(str(i) + "要存的字符是" + c)
    image = create_image_one_char(c, color, background)
    image = convert2gray(np.array(image))
    cv2.imwrite(path, image)
def create_image_one_char(c, color, background):
    font = ImageFont.truetype(DEFAULT_FONTS, 30)
    im = Image.new('RGBA', (WIDHT, HEIGHT), background)
    drawAvatar = ImageDraw.Draw(im)
    w, h = im.size
    drawAvatar.text((4, -3), c, fill=color, font=font)
    del drawAvatar
    # rotate
    im = im.crop(im.getbbox())
    im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)
    # warp
    dx = w * random.uniform(0.1, 0.4)
    dy = h * random.uniform(0.2, 0.5)
    x1 = int(random.uniform(-dx, dx))
    y1 = int(random.uniform(-dy, dy))
    x2 = int(random.uniform(-dx, dx))
    y2 = int(random.uniform(-dy, dy))
    w2 = w + abs(x1) + abs(x2)
    h2 = h + abs(y1) + abs(y2)
    data = (
        x1, y1,
        -x1, h2 - y2,
        w2 + x2, h2 + y2,
        w2 - x2, -y1,
    )
    im = im.resize((w2, h2))
    im = im.transform((WIDHT, HEIGHT), Image.QUAD, data)
    image = Image.new('RGB', (WIDHT, HEIGHT), background)
    image.paste(im, (0, 0), im)
    return image
train_label_name = "code_train_text.txt"
test_label_name = "code_test_text.txt"
def write_image(label_name, dir_path):
    code_list = get_content_from_file(label_name)
    for each in range(len(code_list)):
        generate_image(each, code_list[each], dir_path)
def main():
    write_image(train_label_name, image_path_train)
    write_image(test_label_name, image_path_test)
if __name__ == '__main__':
    main()