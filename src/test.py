import cv2
import os

DIR = "./recording/images"

# downscale = 2 all the images in the directory
downscale = 2


os.mkdir(DIR + "_2")


def downscale_images():
    for filename in os.listdir(DIR):
        img = cv2.imread(os.path.join(DIR, filename))
        img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale))

        cv2.imwrite(os.path.join(DIR + "_2", filename), img)
        print(os.path.join(DIR + "_2", filename))


downscale_images()
