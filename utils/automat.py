import json
import cv2
import os
import matplotlib.pyplot as plt

# Пути к файлам
dataset_path = "C:/ConveyorVision/Dataset/train/"
coco_json_path = os.path.join(dataset_path, "_annotations.coco.json")

# Загружаем COCO-аннотации
with open(coco_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Список классов
categories = {cat["id"]: cat["name"] for cat in data["categories"]}

# Список изображений
image_list = data["images"]
index = 0  # Начинаем с первого изображения

def show_image(idx):
    """Функция для отображения изображения с аннотациями"""
    global index
    index = idx % len(image_list)  # Зацикливаем список изображений
    img_data = image_list[index]
    img_path = os.path.join(dataset_path, img_data["file_name"])
    image = cv2.imread(img_path)

    if image is None:
        print(f"⚠️ Файл {img_data['file_name']} не найден!")
        return

    # Очищаем окно консоли
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"📸 Изображение: {img_data['file_name']}")
    print("🔍 Обнаруженные классы:")

    # Рисуем рамки и подписываем классы
    for ann in data["annotations"]:
        if ann["image_id"] == img_data["id"]:
            x, y, w, h = map(int, ann["bbox"])
            class_id = ann["category_id"]
            class_name = categories[class_id]
            
            # Отображаем рамку и текст (ID + название)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{class_id}: {class_name}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Выводим в консоль
            print(f" - ID {class_id}: {class_name}")

    # Отображение через Matplotlib (работает в Windows)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Изображение {index + 1}/{len(image_list)}: {img_data['file_name']}")
    plt.axis("off")
    plt.show()

# Показываем первое изображение
show_image(index)

while True:
    key = input("➡️ Нажмите 'Enter' для следующего, 'q' для выхода: ")
    if key.lower() == "q":
        print("🚪 Выход")
        break
    index += 1
    show_image(index)
