
import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 1) #1fps (even that my pc cannot do)
cap.set(3, 640)  #640x480 pixels resolution
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #greyscale to reduce channels and load
    _, bw_frame = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)

    img_tensor = transform(bw_frame).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    for idx, box in enumerate(outputs[0]['boxes']): 
        if outputs[0]['scores'][idx] > 0.8:  #if confidence score of model is more than 80%, then draw a box for the bounding area
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Mask R-CNN Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
