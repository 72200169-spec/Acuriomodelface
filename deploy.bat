@echo off
echo 🚀 Iniciando deploy...

echo 1. Verificando Git status...
git status

echo 2. Añadiendo cambios...
git add .

echo 3. Haciendo commit...
git commit -m "Mejora en El Filósofo Virtual - Modos Emocional y Gestos"

echo 4. Subiendo a GitHub...
git push

echo ✅ Deploy completado!
echo Streamlit Community Cloud se actualizará automáticamente en unos minutos.
pause
