import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from model_information import ModelInfo
import tempfile
import pandas as pd
import datetime

# Настройки приложения
st.set_page_config(page_title="Conveyor Vision", layout="centered")

# Загрузка модели
MODEL_PATH = "C:/ConveyorVision/runs/detect/train5/weights/best.pt"
model = YOLO(MODEL_PATH)

# Инициализация журнала событий
if "event_log" not in st.session_state:
    st.session_state.event_log = []

# Инициализация настроек модели
if "detection_threshold" not in st.session_state:
    st.session_state.detection_threshold = 0.5  # По умолчанию 0.5

if "selected_classes" not in st.session_state:
    st.session_state.selected_classes = [0, 1]  # По умолчанию оба класса

# Функция детекции
def detect_objects(image, threshold, classes):
    """Обрабатывает изображение через YOLO и возвращает результаты с учетом порога и классов."""
    results = model(image)
    filtered_results = []
    for result in results:
        filtered_boxes = [box for box in result.boxes if box.conf[0].item() >= threshold and int(box.cls[0].item()) in classes]
        result.boxes = filtered_boxes
        filtered_results.append(result)
    return filtered_results

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

# Функция добавления записи в журнал
def log_event(action, image_name, damage_detected):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event = {
        "Timestamp": timestamp,
        "Action": action,
        "Image Name": image_name,
        "Damage Detected": damage_detected
    }
    st.session_state.event_log.append(event)

# Заголовок приложения
st.markdown("<h1 style='text-align: center;'> Conveyor Vision</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Автоматическое обнаружение повреждений конвейерной ленты</h3>", unsafe_allow_html=True)

# Форма загрузки изображения
uploaded_file = st.file_uploader("🔹 **Выберите изображение**", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Загруженное изображение", use_container_width=True)

    # Панель с настройками модели
    st.sidebar.header("Настройки модели")
    st.sidebar.markdown("Измените порог обнаружения объектов и выберите классы для детекции.")

    # Слайдер для изменения порога обнаружения объектов
    st.session_state.detection_threshold = st.sidebar.slider("Порог обнаружения объектов", 0.0, 1.0, st.session_state.detection_threshold)

    # Чекбоксы для выбора классов для детекции
    class_labels = {0: "Bad", 1: "Good"}
    selected_classes = []
    for class_id, class_label in class_labels.items():
        if st.sidebar.checkbox(class_label, value=(class_id in st.session_state.selected_classes)):
            selected_classes.append(class_id)
    st.session_state.selected_classes = selected_classes

    # Кнопка для сохранения настроек
    if st.sidebar.button("💾 Сохранить настройки"):
        st.success("Настройки сохранены.")

    if st.button("🔍 Запустить детекцию"):
        with st.spinner("⏳ Анализ изображения..."):
            results = detect_objects(image, st.session_state.detection_threshold, st.session_state.selected_classes)
            output_img, damage_detected = draw_results(image, results)
        
        # Сохранение обработанного изображения во временной директории
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            temp_filename = tmp_file.name
            output_pil_img = Image.fromarray(output_img)
            output_pil_img.save(temp_filename)
        
        st.image(output_img, caption="Результат детекции", use_container_width=True)

        # Вывод сообщения в зависимости от наличия повреждений
        if damage_detected:
            st.error(" **Обнаружено повреждение ленты!**", icon="🚨")
        else:
            st.success(" **Лента в нормальном состоянии.**", icon="🟢")

        # Добавление записи в журнал
        log_event("Image processed", uploaded_file.name, damage_detected)

        # Кнопка для скачивания обработанного изображения
        st.download_button(
            label="💾 Сохранить изображение",
            data=open(temp_filename, "rb").read(),
            file_name="detected_image.png",
            mime="image/png"
        )

# Кнопка для отображения журнала событий
if st.button("📜 Показать журнал событий"):
    if st.session_state.event_log:
        event_log_df = pd.DataFrame(st.session_state.event_log)
        st.dataframe(event_log_df)
    else:
        st.write("Журнал событий пуст.")

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
        st.write(f"{key}: {value}")
