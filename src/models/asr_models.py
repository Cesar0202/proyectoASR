import os
import uuid
import speech_recognition as sr
import whisper
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import soundfile as sf
from vosk import Model, KaldiRecognizer
import json

class ASRManager:
    # Esta clase maneja los 4 modelos
    def __init__(self, model_type="whisper_base"):
        self.model_type = model_type
        
    def transcribir(self, audio_path):
        # Funcion principal para pasar de audio a texto
        if self.model_type == "whisper_base":
            return self._transcribir_whisper(audio_path)
        elif self.model_type == "google":
            return self._transcribir_google(audio_path)
        elif self.model_type == "vosk":
            return self._transcribir_vosk(audio_path)
        elif self.model_type == "wav2vec2":
            return self._transcribir_wav2vec2(audio_path)
        return ""

    def _transcribir_whisper(self, audio_path):
        # whisper
        modelo = whisper.load_model("base")
        res = modelo.transcribe(audio_path, language="es")
        return res["text"]

    def _transcribir_google(self, audio_path):
        # google
        r = sr.Recognizer()
        # Convertir a wav temporal usando librosa para evitar dependencia de ffprobe
        temp_wav = f"temp_{uuid.uuid4()}.wav"
        speech, _ = librosa.load(audio_path, sr=16000)
        sf.write(temp_wav, speech, 16000)
        
        with sr.AudioFile(temp_wav) as source:
            data = r.record(source)
            try:
                res = r.recognize_google(data, language="es-ES")
            except:
                res = ""
        
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return res

    def _transcribir_vosk(self, audio_path):
        #vosk
        ruta_modelo = "models/vosk-model-es"
        if not os.path.exists(ruta_modelo):
            return "Error: No esta el modelo de Vosk"
            
        modelo = Model(ruta_modelo)
        
        # Preparar el audio usando librosa
        temp_raw = f"temp_{uuid.uuid4()}.wav"
        speech, _ = librosa.load(audio_path, sr=16000)
        sf.write(temp_raw, speech, 16000, subtype='PCM_16')
        
        rec = KaldiRecognizer(modelo, 16000)
        with open(temp_raw, "rb") as f:
            f.read(44) # saltar cabecera wav
            while True:
                data = f.read(4000)
                if len(data) == 0: break
                rec.AcceptWaveform(data)
        
        res = json.loads(rec.FinalResult())
        if os.path.exists(temp_raw): os.remove(temp_raw)
        return res.get("text", "")

    def _transcribir_wav2vec2(self, audio_path):
        # Meta: wav2vec2
        nombre = "facebook/wav2vec2-large-xlsr-53-spanish"
        proc = Wav2Vec2Processor.from_pretrained(nombre)
        mod = Wav2Vec2ForCTC.from_pretrained(nombre)
        
        # Cargar el audio
        speech, rate = librosa.load(audio_path, sr=16000)
        input_values = proc(speech, return_tensors="pt", sampling_rate=16000).input_values
        
        with torch.no_grad():
            logits = mod(input_values).logits
        
        ids = torch.argmax(logits, dim=-1)
        res = proc.batch_decode(ids)[0]
        return res.lower()
