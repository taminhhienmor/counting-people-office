from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
import argparse

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detector(image):
    clone = image.copy()

    (rects, weights) = hog.detectMultiScale(image, winStride=(1, 1), padding=(60, 60), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    result = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    return result

def argsParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=None, help="path to images directory")
    # args = vars(ap.parse_args())
    args = {'image': 'test.png'}

    return args


def localDetect(image_path):
    result = []
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    clone = image.copy()
    if len(image) <= 0:
        print("[ERROR] could not read your local image")
        return result
    print("[INFO] Detecting people")
    result = detector(image)

    # shows the result
    index = 1
    for (xA, yA, xB, yB) in result:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.putText(image, "{}".format(index), (xA + 5, yA + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        index += 1

    # cv2.imshow("result", image)
    print("Counting people: {}".format(index))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("result.png", np.hstack((clone, image)))
    # return (result, image)
    return "Counting people: {}".format(index)

def detectPeople(args):
    image_path = args["image"]
    if image_path != None:
        print("[INFO] Image path provided, attempting to read image")
        # (result, image) = localDetect(image_path)
        return localDetect(image_path)

def main():
    args = argsParser()
    # detectPeople(args)
    return detectPeople(args)

if __name__ == '__main__':
    main()