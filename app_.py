import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os

# Настройки интерфейса
st.set_page_config(page_title="Conveyor Vision", layout="centered")

# Загрузка модели
MODEL_PATH = "runs/detect/train5/weights/best.pt"  # Укажи путь, если модель лежит в другом месте
model = YOLO(MODEL_PATH)

# Обработка изображения через YOLO
def detect_objects(image):
    results = model(image)
    return results

# Отрисовка результатов через PIL
def draw_results_pil(image, results):
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    damage_detected = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]}: {conf:.2f}"

            if cls == 0:
                color = (255, 0, 0)  # Красный для "Bad"
                damage_detected = True
            else:
                color = (0, 255, 0)  # Зелёный для "Good"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 10), label, fill=color, font=font)

    return img, damage_detected

# Заголовки
st.title("🔍 Conveyor Vision")
st.markdown("Автоматическое определение повреждений конвейерной ленты")

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    if st.button("Запустить детекцию"):
        with st.spinner("Анализ изображения..."):
            results = detect_objects(image)
            output_img, damage_detected = draw_results_pil(image, results)

        st.image(output_img, caption="Результат", use_container_width=True)

        if damage_detected:
            st.error("🚨 Обнаружено повреждение ленты!")
        else:
            st.success("🟢 Повреждений не обнаружено")

        # Скачивание
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            output_img.save(tmp.name)
            st.download_button("💾 Скачать изображение", data=open(tmp.name, "rb"), file_name="result.png", mime="image/png")
            os.unlink(tmp.name)
