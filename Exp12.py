import cv2 as cv

# Models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

mean = (78.426, 87.769, 114.896)

# Load models
faceNet = cv.dnn.readNet(faceModel, faceProto)
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)

# Read image
img = cv.imread("2.jpg")

# Detect face
blob = cv.dnn.blobFromImage(img, 1.0, (300, 300),
                            [104, 117, 123], True, False)
faceNet.setInput(blob)
detections = faceNet.forward()

for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > 0.7:
        h, w = img.shape[:2]

        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        face = img[y1:y2, x1:x2]

        # Prepare face
        blob = cv.dnn.blobFromImage(
            face, 1.0, (227, 227), mean, swapRB=False
        )

        # Gender
        genderNet.setInput(blob)
        gender = genderList[genderNet.forward()[0].argmax()]

        # Age
        ageNet.setInput(blob)
        age = ageList[ageNet.forward()[0].argmax()]

        print("Gender:", gender)
        print("Age:", age)

        cv.rectangle(img, (x1, y1), (x2, y2),
                     (0, 255, 0), 2)

        cv.putText(img, gender + ", " + age,
                   (x1, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 255, 255), 2)

cv.imshow("Result", img)
cv.waitKey(0)
cv.destroyAllWindows()
