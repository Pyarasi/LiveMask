import cv2
import torch
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms as T

model = fcn_resnet50(weights="FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((720, 1280)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor()
])

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('data/cascades/Mouth Cascade.xml')
nose_cascade = cv2.CascadeClassifier('data/cascades/Nose Cascade.xml')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 3)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    noses = nose_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    mouths = mouth_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        cv2.rectangle(gray_frame, (x, y), (x+w, y+h), (200, 200, 200), 2)

    for (x, y, w, h) in noses:
        cv2.rectangle(gray_frame, (x, y), (x+w, y+h), (150, 150, 150), 2)

    for (x, y, w, h) in mouths:
        cv2.rectangle(gray_frame, (x, y), (x+w, y+h), (100, 100, 100), 2)

    cv2.imshow('Live Webcam Detection', gray_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()