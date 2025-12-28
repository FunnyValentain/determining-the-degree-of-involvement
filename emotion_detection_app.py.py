import streamlit as st
import requests
import time
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np
from PIL import Image

# Настройки
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TITLE = "🎭 Emotion Detection from Video"
ICON = "🎭"

# Конфигурация страницы
st.set_page_config(
    page_title=TITLE,
    page_icon=ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
    .processing-msg {
        color: #ffc107;
        font-weight: bold;
    }
    .video-container {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown(f'<h1 class="main-header">{TITLE}</h1>', unsafe_allow_html=True)
    
    # Сайдбар с информацией
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1077/1077063.png", width=100)
        st.markdown("### ℹ️ О проекте")
        st.markdown("""
        Эта система определяет эмоции людей на видео:
        
        - 📹 Загрузите видео файл
        - ⚡ Система обработает его
        - 🎭 Определит эмоции на каждом кадре
        - 📥 Скачайте результат
        
        Поддерживаемые форматы: MP4, AVI, MOV
        Максимальный размер: 100MB
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Технологии")
        st.markdown("""
        - MediaPipe для детекции лиц
        - EmotiEffLib для распознавания эмоций
        - Streamlit для интерфейса
        - FastAPI для бэкенда
        """)
    
    # Основной контент
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Загрузка видео")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите видео файл",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Поддерживаются MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None:
            # Показываем информацию о файле
            file_details = {
                "Имя файла": uploaded_file.name,
                "Тип файла": uploaded_file.type,
                "Размер файла": f"{uploaded_file.size / (1024*1024):.2f} MB"
            }
            
            st.json(file_details)
            
            # Предпросмотр видео
            st.markdown("### 👀 Предпросмотр")
            
            # Сохраняем временный файл для предпросмотра
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            temp_file.close()
            
            # Показываем видео
            video_bytes = uploaded_file.read()
            st.video(video_bytes)
            
            # Кнопка обработки
            if st.button("🚀 Начать обработку", type="primary", use_container_width=True):
                with st.spinner("Отправка видео на обработку..."):
                    try:
                        # Отправляем файл на бэкенд
                        files = {"video": (uploaded_file.name, video_bytes, uploaded_file.type)}
                        response = requests.post(f"{BACKEND_URL}/upload", files=files)
                        
                        if response.status_code == 200:
                            data = response.json()
                            task_id = data["task_id"]
                            st.session_state.task_id = task_id
                            st.session_state.status = "processing"
                            st.success("✅ Видео успешно загружено! Идет обработка...")
                        else:
                            st.error(f"❌ Ошибка загрузки: {response.text}")
                            
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)}")
    
    with col2:
        st.markdown("### 📊 Статус обработки")
        
        if "task_id" in st.session_state:
            task_id = st.session_state.task_id
            
            # Периодически проверяем статус
            if st.button("🔄 Проверить статус", use_container_width=True):
                try:
                    response = requests.get(f"{BACKEND_URL}/status/{task_id}")
                    
                    if response.status_code == 200:
                        status_data = response.json()
                        current_status = status_data["status"]
                        
                        st.info(f"Статус: **{current_status}**")
                        
                        if current_status == "completed":
                            st.session_state.status = "completed"
                            st.session_state.result_path = status_data["result_path"]
                            st.success("✅ Обработка завершена!")
                            
                            # Кнопка скачивания
                            download_url = f"{BACKEND_URL}/download/{task_id}"
                            st.markdown(f"""
                            ### 📥 Скачать результат
                            [Нажмите чтобы скачать]({download_url})
                            """)
                            
                        elif current_status == "failed":
                            st.error(f"❌ Ошибка обработки: {status_data.get('error', 'Неизвестная ошибка')}")
                            
                        else:
                            st.warning("⏳ Обработка еще не завершена...")
                            
                    else:
                        st.error("❌ Не удалось получить статус")
                        
                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
            
            # Прогресс бар (заглушка)
            if st.session_state.get("status") == "processing":
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.1)
                    progress_bar.progress(i + 1)
    
    # Пример результата
    st.markdown("---")
    st.markdown("### 🎭 Пример результата")
    
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        st.markdown("**Детекция лиц**")
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*ZCjPUFrB6eHRI7-a3XBNdQ.jpeg", 
                caption="Обнаружение лиц и ключевых точек")
    
    with col_ex2:
        st.markdown("**Распознавание эмоций**")
        st.image("https://viso.ai/wp-content/uploads/2021/05/facial-expression-recognition-software.png",
                caption="Определение эмоций по выражению лица")
    
    with col_ex3:
        st.markdown("**Результат обработки**")
        st.image("https://www.researchgate.net/profile/Amir-Hussain-8/publication/327404470/figure/fig3/AS:668258825682954@1536341716485/Sample-output-of-emotion-detection-on-video-frame-sequence.ppm",
                caption="Видео с аннотациями эмоций")

def check_backend_connection():
    """Проверяет подключение к бэкенду"""
    try:
        response = requests.get(f"{BACKEND_URL}/docs", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    # Проверка подключения к бэкенду
    if not check_backend_connection():
        st.warning("⚠️ Бэкенд не доступен. Убедитесь что сервер запущен.")
        st.info(f"Ожидаемый URL бэкенда: {BACKEND_URL}")
        
        if st.button("🔄 Проверить снова"):
            st.rerun()
    else:
        main()
