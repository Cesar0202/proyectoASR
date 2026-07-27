# ASR Lab - Grupo 3

Este es nuestro proyecto para evaluar qué tan bien funcionan los sistemas de reconocimiento de voz (ASR) cuando las personas tartamudean o tienen disfluencias en el habla.

### ¿De qué trata?
Básicamente, agarramos varios modelos de IA (Google, Vosk, Whisper y Wav2Vec2) y les pasamos audios con diferentes tipos de problemas: repeticiones de palabras, pausas largas, sonidos de relleno (como "eee", "este"), etc. 

El objetivo es ver cuál de todos es más "robusto" y mantiene mejor el mensaje original a pesar de los fallos en el habla.

### ¿Qué tiene la app?
*   **Dashboard:** Gráficos comparativos de error (WER y CER).
*   **Pruebas en vivo:** Puedes subir tu propio audio o usar uno del sistema para ver cómo lo transcribe cada modelo.
*   **Análisis de errores:** Una sección donde explicamos por qué los modelos fallan (si borran palabras o se las inventan).

### Cómo hacerlo correr
1. Instala las librerías necesarias:
   ```bash
   pip install -r requirements.txt
   ```
2. Lanza la aplicación:
   ```bash
   streamlit run app.py
   ```

### Notas del Grupo
*   Las métricas ignoran tildes y comas para que la comparación sea justa (basada en el sonido y no en la ortografía).
*   Si lo corres localmente, asegúrate de tener los modelos en la carpeta `/models`.

---
**Integrantes - Grupo 3**
*   Huaman Uriarte Cesar Alberto
*   Ccora Quispe Holiver Jhunior
*   Liñan Paredes Saul Alexander
*   Cespedes Viguria Jhamir Sebastian
