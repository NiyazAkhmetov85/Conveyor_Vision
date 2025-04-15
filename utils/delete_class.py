import json

# Загрузка файла
annotations_path = "C:/ConveyorVision/Dataset/valid/_annotations.coco.json"

with open(annotations_path, "r") as f:
    data = json.load(f)

# Убираем ID 0 из категорий
data["categories"] = [cat for cat in data["categories"] if cat["id"] != 0]

# Сохраняем исправленный файл
with open(annotations_path, "w") as f:
    json.dump(data, f, indent=4)

print("✅ Класс ID 0 ('Good') удален. Датасет исправлен!")
