# ASR Lab - Grupo 3

Este es nuestro proyecto para evaluar qué tan bien funcionan los sistemas de reconocimiento de voz (ASR) cuando las personas tartamudean o tienen disfluencias en el habla.

### ¿De qué trata?
Básicamente, agarramos varios modelos de IA (Google, Vosk, Whisper y Wav2Vec2) y les pasamos audios con diferentes tipos de problemas: repeticiones de palabras, pausas largas, sonidos de relleno (como "eee", "este"), etc. 

El objetivo es ver cuál de todos es más "robusto" y mantiene mejor el mensaje original a pesar de los fallos en el habla.

### Estructura del Proyecto
*   `app.py`: Archivo principal de la aplicación Streamlit que contiene la interfaz web y el dashboard interactivo.
*   `arreglar_todo.py`: Script auxiliar utilizado para corregir o procesar archivos de manera masiva.
*   `models/`: Carpeta donde se deben ubicar los modelos locales. Por ejemplo, aquí debe ir la carpeta del modelo de Vosk (ej. `vosk-model-es`).
*   `results/`: Contiene los archivos generados tras las evaluaciones, audios, metadatos (`metadata.json`) y resultados en formato JSON (`results_final.json`).
*   `src/`: Código fuente con la lógica interna. Incluye scripts para evaluación (`evaluator.py`), modelos y ejecución de experimentos (`run_experiments.py`).
*   `requirements.txt`: Archivo con las dependencias necesarias para correr el proyecto.

### ¿Qué tiene la app?
*   **Dashboard:** Gráficos comparativos de error (WER y CER).
*   **Pruebas en vivo:** Puedes subir tu propio audio o usar uno del sistema para ver cómo lo transcribe cada modelo.
*   **Análisis de errores:** Una sección donde explicamos por qué los modelos fallan (si borran palabras o se las inventan).

### Cómo hacerlo correr
1. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```
2. Instala las librerías necesarias:
   ```bash
   pip install -r requirements.txt
   ```
3. Descarga el modelo local de Vosk:
   * Ve a la página oficial de modelos de Vosk: [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
   * Descarga un modelo en español (por ejemplo, `vosk-model-es-0.42`).
   * Descomprime el archivo descargado y pon la carpeta resultante dentro de la carpeta `models/` de este proyecto, de manera que la estructura quede así: `models/vosk-model-es/`.
4. Lanza la aplicación:
   ```bash
   streamlit run app.py
   ```

### Notas del Grupo
*   Las métricas ignoran tildes y comas para que la comparación sea justa (basada en el sonido y no en la ortografía).
*   Si lo corres localmente, asegúrate de tener los modelos descargados en la carpeta `/models`.

---
**Integrantes - Grupo 3**
*   Huaman Uriarte Cesar Alberto
*   Ccora Quispe Holiver Jhunior
*   Liñan Paredes Saul Alexander
*   Cespedes Viguria Jhamir Sebastian
