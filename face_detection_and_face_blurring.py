import numpy as np
import cv2

haarcascade = cv2.CascadeClassifier("haarcascade_profileface.xml")

image_paths = ['musk_full_body.jpg', 'musk_other_object.jpg', 'musk_with_other.jpg']


for image_name in image_paths:
    
    image = cv2.imread(image_name)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get faces
    faces = haarcascade.detectMultiScale(gray)

    for face in faces:
        x, y, w, h = face
        image[y : y + h, x : x + w] = cv2.GaussianBlur(image[y : y + h, x : x + w], (25, 25), 0)
        
        steps_count = 15
        stepsX = np.linspace(x, x + w, steps_count + 1,dtype = np.int)
        stepsY = np.linspace(y, y + h, steps_count + 1,dtype = np.int)
        
        for direction_x in range(steps_count):
            for direction_y in range(steps_count):
                color = cv2.mean(
                    image[
                        stepsY[direction_y] : stepsY[direction_y + 1],
                        stepsX[direction_x] : stepsX[direction_x + 1],
                    ]
                )
                # image[y: y + h, steps[i]] : steps[i + 1]
                cv2.rectangle(
                    image,
                    (stepsX[direction_x], stepsY[direction_y]),
                    (stepsX[direction_x + 1], stepsY[direction_y + 1]),
                    color, 
                    -1)

cv2.imshow("Image", image)
cv2.waitKey(5000)