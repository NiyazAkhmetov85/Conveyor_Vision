import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from model_information import ModelInfo

# Настройки приложения
st.set_page_config(page_title="Conveyor Vision", layout="centered")

# Загрузка модели
MODEL_PATH = "C:/ConveyorVision/runs/detect/train5/weights/best.pt"
model = YOLO(MODEL_PATH)

# Функция детекции
def detect_objects(image):
    """Обрабатывает изображение через YOLO и возвращает результаты."""
    results = model(image)
    return results

# Функция отрисовки результатов
def draw_results(image, results):
    """Отрисовывает предсказанные объекты и возвращает изображение."""
    img = np.array(image)
    damage_detected = False
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]}: {conf:.2f}"
            
            # Определяем цвет в зависимости от класса
            if cls == 0:  # "Bad" (повреждение)
                color = (0, 0, 255)  # Красный
                damage_detected = True
            else:  # "Good" (норма)
                color = (0, 255, 0)  # Зеленый

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img, damage_detected

# Заголовок приложения
st.markdown("<h1 style='text-align: center;'> Conveyor Vision</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Автоматическое обнаружение повреждений конвейерной ленты</h3>", unsafe_allow_html=True)

# Форма загрузки изображения
uploaded_file = st.file_uploader("🔹 **Выберите изображение**", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Загруженное изображение", use_container_width=True)

    if st.button("🔍 Запустить детекцию"):
        with st.spinner("⏳ Анализ изображения..."):
            results = detect_objects(image)
            output_img, damage_detected = draw_results(image, results)
        
        st.image(output_img, caption="Результат детекции", use_container_width=True)

        # Вывод сообщения в зависимости от наличия повреждений
        if damage_detected:
            st.error(" **Обнаружено повреждение ленты!**", icon="🚨")
        else:
            st.success(" **Лента в нормальном состоянии.**", icon="🟢")

# Инициализация состояния кнопки
if "show_model_info" not in st.session_state:
    st.session_state.show_model_info = False

# Обработчик нажатия кнопки
if st.button("ℹ️ Информация о модели"):
    st.session_state.show_model_info = not st.session_state.show_model_info

# Отображение или скрытие информации о модели
if st.session_state.show_model_info:
    model_info = ModelInfo()
    info = model_info.info
    st.markdown("### Информация о модели")
    for key, value in info.items():
        st.write(f"**{key}**: {value}")
