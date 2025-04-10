from ultralytics import YOLO

# Загружаем модель
model = YOLO("yolov8n.pt")

# Запускаем обучение
model.train(data="C:/ConveyorVision/Dataset/data.yaml", epochs=50, imgsz=640, batch=16)
