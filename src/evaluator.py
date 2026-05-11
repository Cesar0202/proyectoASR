import json
import os
import sys
import jiwer
import shutil
from datetime import datetime

# Para que encuentre los modelos
sys.path.append(os.path.join(os.getcwd(), "src"))
from models.asr_models import ASRManager

class Evaluator:
    #clase para calificar los audios
    def __init__(self, metadata_path="results/metadata.json"):
        self.metadata_path = metadata_path
        self.ruta_salida = "results/results_final.json"
        self.carpeta_backups = "results/backups"
        if not os.path.exists(self.carpeta_backups):
            os.makedirs(self.carpeta_backups)
        
    def _hacer_backup(self):
        #guarda una copia por si borramos algo sin querer
        if os.path.exists(self.ruta_salida):
            fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre = f"copia_{fecha}.json"
            shutil.copy2(self.ruta_salida, os.path.join(self.carpeta_backups, nombre))

    def correr_evaluacion(self, modelos=["whisper_base"]):
        # Esta funcion hace todo el trabajo de calificar
        if not os.path.exists(self.metadata_path):
            return

        # Primero hacemos la copi
        self._hacer_backup()
            
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            muestras_nuevas = json.load(f)
            
        #cargamos lo que ya tenemos guardado
        resultados_finales = {}
        if os.path.exists(self.ruta_salida):
            try:
                with open(self.ruta_salida, "r", encoding="utf-8") as f:
                    resultados_finales = json.load(f)
            except:
                pass

        for nombre_modelo in modelos:
            # Empezamos con cada modelo
            if nombre_modelo not in resultados_finales:
                resultados_finales[nombre_modelo] = []
            
            # Ver que no esten repetidos
            ya_estaban = {r['id'] for r in resultados_finales[nombre_modelo]}
            
            # Solo los que faltan
            por_hacer = [m for m in muestras_nuevas if m['id'] not in ya_estaban]
            
            if not por_hacer:
                continue
                
            asr = ASRManager(model_type=nombre_modelo)
            
            for i, muestra in enumerate(por_hacer):
                # Mensaje de progreso simple
                if i % 10 == 0:
                    print(f"Modelo {nombre_modelo}: {i}/{len(por_hacer)}")
                
                try:
                    # Transcribir y comparar
                    prediccion = asr.transcribir(muestra['audio_path'])
                    real = muestra['text_original'].lower()
                    pred = prediccion.lower()
                    
                    calc = jiwer.process_words(real, pred)
                    
                    # Guardamos los datos
                    info = muestra.copy()
                    info["model"] = nombre_modelo
                    info["prediction"] = prediccion
                    info["wer"] = calc.wer
                    info["cer"] = jiwer.cer(real, pred)
                    info["substitutions"] = calc.substitutions
                    info["deletions"] = calc.deletions
                    info["insertions"] = calc.insertions
                    
                    if "control" in muestra['audio_path']:
                        info["disfluency_type"] = "control"
                        
                    resultados_finales[nombre_modelo].append(info)
                except:
                    pass

            # Guardar el archivo al final de cada modelo
            with open(self.ruta_salida, "w", encoding="utf-8") as f:
                json.dump(resultados_finales, f, indent=4, ensure_ascii=False)
