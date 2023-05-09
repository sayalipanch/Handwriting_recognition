import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model
model = load_model('handwritten_character_recog_model.h5')

words = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

image = cv2.imread('R.png')
image_copy = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (400,440))

image_copy = cv2.GaussianBlur(image_copy, (7,7), 0)
gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

# def sort_contours(cnts, method="left-to-right"):
#     reverse = False
#     i = 0
#     if method == "right-to-left" or method == "bottom-to-top":
#         reverse = True
#     if method == "top-to-bottom" or method == "bottom-to-top":
#         i = 1
#     boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#     (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
#     return (cnts, boundingBoxes)

# def get_letters(img):
#     letters = []
#     image = cv2.imread(img)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#     dilated = cv2.dilate(thresh1, None, iterations=2)
#     cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     cnts = sort_contours(cnts, method="left-to-right")[0]
#     for c in cnts:
#         if cv2.contourArea(c) > 10:
#             (x, y, w, h) = cv2.boundingRect(c)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         roi = gray[y:y + h, x:x + w]
#         thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#         thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
#         thresh = thresh.astype("float32") / 255.0
#         thresh = np.expand_dims(thresh, axis=-1)
#         thresh = thresh.reshape(1, 32, 32, 1)
#         ypred = model.predict(thresh)
#         print(ypred)
#         # ypred = LB.inverse_transform(ypred)
#         [x] = ypred
#         letters.append(x)
#     return letters, image


#----- added
# def apply_spatial_filter(image, kernel):
#     # Apply the spatial filter
#     filtered_image = cv2.filter2D(image, -1, kernel)
    
#     return filtered_image

# # Create a sharpening filter (3x3)
# sharpening_kernel = np.array([[0, -1, 0],
#                               [-1, 4, -1],
#                               [0, -1, 0]], np.float32)

# # Apply the sharpening filter
# sharpened_image = apply_spatial_filter(gray_image, sharpening_kernel)


#-----
_, img_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)

final_image = cv2.resize(img_thresh, (28,28))
final_image =np.reshape(final_image, (1,28,28,1))

prediction = words[np.argmax(model.predict(final_image))]

cv2.putText(image, "Prediction: " + prediction, (20,410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (0,255,0))
cv2.imshow('Project Gurukul handwritten character recognition ', image)

while (1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
