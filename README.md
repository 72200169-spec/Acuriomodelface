# Reconocimiento de Expresiones Faciales

Aplicación web interactiva para reconocer expresiones faciales usando TensorFlow, OpenCV y Streamlit.

## Funciones
- 📷 Subir una imagen y analizar la expresión facial
- 🎥 Usar la cámara web para tomar una foto y analizarla
- Muestra la emoción detectada con emoji
- Muestra gráfico de probabilidades de cada emoción

## Emociones detectadas
- 😠 Angry (Enojado)
- 🤢 Disgust (Disgustado)
- 😨 Fear (Asustado)
- 😊 Happy (Feliz)
- 😐 Neutral (Neutral)
- 😢 Sad (Triste)
- 😮 Surprise (Sorprendido)

## Cómo usar
1. Clona este repositorio
2. Instala las dependencias: `pip install -r requirements.txt`
3. Ejecuta la app: `streamlit run app.py`
4. Usa la opción de subir imagen o la cámara web

## Archivos importantes
- `train.py`: Script para entrenar el modelo
- `app.py`: Aplicación Streamlit
- `mejor_modelo.keras`: Modelo preentrenado (ya está incluido!)
- `requirements.txt`: Lista de dependencias
