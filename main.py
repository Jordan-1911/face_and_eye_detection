import cv2
import mediapipe as mp

# read image
# image_paths = ['musk_costume1.jpg', 'musk_distant.jpg', 'musk_full_body.jpg', 'musk_other_object.jpg', 'multiple_faces.jpg']
img_path = 'musk_costume1.jpg'

img = cv2.imread(img_path)

# detect faces
mp_face_detection = mp.solutions.face_detection

# 0 means faces that are close to the camera, 1 for larger distances
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    # loop through each detected face
    for detection in results.detections:
        # extract the bounding box of the face
        bbox = detection.location_data.relative_bounding_box
        h, w, c = img.shape
        x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

        # extract the face region
        face = img[y:y+bh, x:x+bw]

        # detect eyes in the face region using Haar cascades
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # blur each detected eye
        for (ex, ey, ew, eh) in eyes:
            # extract the eye region
            eye_roi = face[ey:ey+eh, ex:ex+ew]

            # apply a Gaussian blur to the eye region
            eye_roi_blurred = cv2.GaussianBlur(eye_roi, (15, 15), 30)

            # replace the eye region with the blurred eye region
            face[ey:ey+eh, ex:ex+ew] = eye_roi_blurred

        # replace the face region with the modified version
        img[y:y+bh, x:x+bw] = face

# display the modified image
cv2.imshow('Blurred Eyes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
