# 📜 El Filósofo Virtual

Proyecto académico de reconocimiento de emociones faciales y gestos de mano con Streamlit.

## Funcionalidades
- 🧘 **Modo Filósofo Emocional**: Detecta tu emoción facial (felicidad, tristeza, enojo, sorpresa, miedo, disgusto, neutral) y muestra una cita filosófica correspondiente de Aristóteles, Nietzsche, Sartre o Kant.
- 🖐️ **Modo Gestos**: Detecta gestos de mano (mano abierta, pulgar arriba, OK) y responde interactivamente.

## Tecnologías
- Streamlit: Interfaz web
- TensorFlow/Keras: Reconocimiento de emociones
- OpenCV: Detección de rostros
- MediaPipe: Reconocimiento de gestos

## Cómo ejecutar localmente
1. Crea y activa un entorno virtual:
   - Windows: `python -m venv venv && .\venv\Scripts\activate`
   - Linux/Mac: `python3 -m venv venv && source venv/bin/activate`

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta la app:
   ```bash
   streamlit run app.py
   ```

## Script de Deploy Automático
Para subir cambios a GitHub y actualizar la app en Streamlit Community Cloud:
1. Asegúrate de tener Git configurado
2. Ejecuta:
   - Windows: `.\deploy.bat`
   - Linux/Mac: `bash deploy.sh`

## Archivos principales
- `train.py`: Entrenamiento del modelo de emociones
- `app.py`: Aplicación principal
- `mejor_modelo.keras`: Modelo pre-entrenado
- `requirements.txt`: Lista de dependencias
