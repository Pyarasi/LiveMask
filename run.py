import cv2
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torchvision.transforms as T

# Step 1: Load the Pre-trained SSD Lite MobileNet
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Transform Function (Grayscale and Resize)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((240, 320)),  # 240p resolution
    T.Grayscale(num_output_channels=1),  # Convert to grayscale
    T.ToTensor()
])

# Step 3: Capture Video (Set to 15 FPS)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)  # Attempt to set frame rate to 15 FPS

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to a smaller resolution to reduce processing time
    frame = cv2.resize(frame, (320, 240))
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Transform and add batch dimension
    img_tensor = transform(gray_frame).unsqueeze(0).to(device)

    with torch.no_grad():
        # Mixed Precision Speedup if using GPU
        with torch.cuda.amp.autocast():
            outputs = model(img_tensor)

    # Draw detections based on the lighter model’s predictions
    for idx, box in enumerate(outputs[0]['boxes']):
        score = outputs[0]['scores'][idx].item()
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(gray_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the processed frame
    cv2.imshow('Live Webcam Detection', gray_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()