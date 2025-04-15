import os
import json
from pathlib import Path

def convert_coco_to_yolo(dataset_folder):
    coco_json_path = os.path.join(dataset_folder, "_annotations.coco.json")  # Исправленный путь к файлу аннотаций
    output_dir = os.path.join(dataset_folder, "labels")
    
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    category_mapping = {1: 0, 2: 1}  # Перенумерация классов
    
    images = {img['id']: img['file_name'] for img in coco_data['images']}
    annotations = coco_data['annotations']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for ann in annotations:
        image_id = ann['image_id']
        bbox = ann['bbox']
        category_id = ann['category_id']
        
        if category_id not in category_mapping:
            continue  # Пропускаем ненужные классы
        
        new_category_id = category_mapping[category_id]
        x_center = (bbox[0] + bbox[2] / 2) / coco_data['images'][image_id-1]['width']
        y_center = (bbox[1] + bbox[3] / 2) / coco_data['images'][image_id-1]['height']
        width = bbox[2] / coco_data['images'][image_id-1]['width']
        height = bbox[3] / coco_data['images'][image_id-1]['height']
        
        annotation_text = f"{new_category_id} {x_center} {y_center} {width} {height}\n"
        
        annotation_filename = Path(output_dir) / f"{Path(images[image_id]).stem}.txt"
        with open(annotation_filename, 'a') as file:
            file.write(annotation_text)
        
        print(f"Аннотация сохранена: {annotation_filename}")

if __name__ == "__main__":
    dataset_folder = "C:/ConveyorVision/Dataset/valid"  # Изменяйте только эту строку (train, valid, test)
    convert_coco_to_yolo(dataset_folder)
    print("✅ Конвертация завершена!")
