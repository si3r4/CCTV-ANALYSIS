import cv2
face_cascade = cv2.CascadeClassifier('./haarcascade_russian_plate_number.xml')
img = cv2.imread('./assets/car.JPG')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, x1, y1) in faces:
    cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
# Display the output
cv2.imshow('img', img)
cv2.waitKey()