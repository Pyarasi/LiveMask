import cv2
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torchvision.transforms as T

model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((120, 160)),  
    T.Grayscale(num_output_channels=1),  
    T.ToTensor()
])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 3)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (160, 120))
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_tensor = transform(gray_frame).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(img_tensor)

    for idx, box in enumerate(outputs[0]['boxes']):
        score = outputs[0]['scores'][idx].item()
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(gray_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Live Webcam Detection', gray_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()