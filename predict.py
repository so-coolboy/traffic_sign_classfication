#导包
import argparse
import cv2
import numpy as np
import imutils
from keras.models import load_model
from keras.preprocessing.image import img_to_array

NORM_SIZE = 32

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to trained model model")
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-s", "--show", required=True, action="store_true", help="show predict image", default=False)
    args = vars(ap.parse_args())
    return args

def predict(args):
    print("loading model...")
    model = load_model(args['model'])

    print("loading image...")
    image = cv2.imread(args['image'])
    orig = image.copy()

    #预处理
    image = cv2.resize(image,(NORM_SIZE, NORM_SIZE))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    #预测
    result = model.predict(image)[0]
    proba = np.max(result)
    label = str(np.where(result==proba)[0])
    label = "{}: {:.2f}%".format(label, proba*100)
    print(label)

    if args['show']:
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Output", output)
        cv2.waitKey(0)


if __name__=="__main__":
    args = args_parse()
    predict(args)

