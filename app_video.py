import streamlit as st
import cv2
import yt_dlp
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Загрузка модели YOLO (укажите путь к вашей модели)
MODEL_PATH = "C:/ConveyorVision/runs/detect/train5/weights/best.pt"
model = YOLO(MODEL_PATH)

# Функция для потоковой обработки видео с YouTube
def process_video_stream(video_url, frame_skip):
    """Обрабатывает потоковое видео с YouTube, извлекая кадры с заданным шагом."""
    ydl_opts = {
        'format': 'best',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        video_url = info_dict.get("url", None)

    cap = cv2.VideoCapture(video_url)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Добавление кадра с заданным шагом
        if count % frame_skip == 0:
            frames.append(frame)

        count += 1

    cap.release()
    return frames

# Функция детекции объектов
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

# Интерфейс пользователя
st.set_page_config(page_title="Conveyor Vision", layout="centered")
st.markdown("<h1 style='text-align: center;'>Conveyor Vision</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Детекция повреждений на конвейерной ленте</h3>", unsafe_allow_html=True)

# Поле для ввода ссылки на видео с YouTube
youtube_url = st.text_input("Введите ссылку на видео с YouTube:")

# Поле для ввода количества кадров для пропуска
frame_skip = st.number_input("Введите количество кадров для пропуска (например, 10):", min_value=1, value=10, step=1)

# Кнопка для обработки видео
if st.button("Обработать видео"):
    if youtube_url:
        with st.spinner("Обработка видео..."):
            frames = process_video_stream(youtube_url, frame_skip)
            st.success("Видео обработано успешно!")

            # Визуализация всех извлеченных кадров
            st.write("**Извлеченные кадры:**")
            for idx, frame in enumerate(frames):
                st.image(frame, caption=f"Кадр {idx}", use_container_width=True)
            
            st.session_state.frames = frames  # Сохраняем кадры в состоянии

    else:
        st.error("Пожалуйста, введите ссылку на видео.")

# Кнопка для запуска детекции
if st.button("Запустить детекцию"):
    if "frames" in st.session_state:
        with st.spinner("Запуск детекции..."):
            detection_threshold = 0.5  # Установим порог детекции
            selected_classes = [0]  # Предположим, что интересует только класс "bad"
            for frame in st.session_state.frames:
                results = detect_objects(frame, detection_threshold, selected_classes)
                output_img, damage_detected = draw_results(frame, results)
                
                if damage_detected:
                    st.error("Обнаружен поврежденный участок ленты", icon="🚨")
                    st.image(output_img, caption="Кадр с повреждением", use_container_width=True)
                    break  # Останавливаемся на первом обнаруженном повреждении
            else:
                st.success("Повреждений не обнаружено.")
    else:
        st.error("Сначала обработайте видео.")

# Добавить здесь остальные функциональные блоки позже
