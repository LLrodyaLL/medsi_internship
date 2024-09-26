import torch
import cv2
import numpy as np
from pathlib import Path

model_path = 'best.pt'
image_path = 'content/val_data/images/6de94847-out4593.png'

model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (704, 576))
img_tensor = torch.from_numpy(img_resized).float() / 255.0
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

with torch.no_grad():
    results = model(img_tensor)

pred = results[0]
for det in pred:
    if det[4] > 0.5:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

cv2.imshow('Predictions', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
