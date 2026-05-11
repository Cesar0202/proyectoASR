import json
import os
import sys
import jiwer
import gc
import torch

# Añadir la carpeta src
sys.path.append(os.path.join(os.getcwd(), "src"))
from models.asr_models import ASRManager

def arreglar():
    ruta_resultados = "results/results_final.json"
    ruta_control = "results/metadata_control.json"
    
    if not os.path.exists(ruta_resultados):
        print("Error: No existe el archivo de resultados.")
        return
    
    with open(ruta_resultados, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    with open(ruta_control, "r", encoding="utf-8") as f:
        meta_control = json.load(f)
        
    modelos = ["google", "vosk", "whisper_base", "wav2vec2"]
    
    print("\n--- INICIANDO REPARACIÓN DE EMERGENCIA ---")
    
    for m_name in modelos:
        print(f"\nTrabajando en: {m_name}")
        if m_name not in data:
            data[m_name] = []
            
        ids_viejos = {r['id'] for r in data[m_name]}
        
        #procesar audios uno por uno
        agregados = 0
        for item in meta_control:
            if item['id'] not in ids_viejos:
                try:
                    asr = ASRManager(m_name)
                    p = asr.transcribir(item['audio_path'])
                    
                    ref = item['text_original'].lower()
                    hyp = p.lower()
                    
                    calc = jiwer.process_words(ref, hyp)
                    res = item.copy()
                    res.update({
                        "model": m_name,
                        "prediction": p,
                        "wer": calc.wer,
                        "cer": jiwer.cer(ref, hyp),
                        "substitutions": calc.substitutions,
                        "deletions": calc.deletions,
                        "insertions": calc.insertions,
                        "disfluency_type": "control"
                    })
                    data[m_name].append(res)
                    agregados += 1
                    print(f"  [OK] {item['id']}")
                    
                    # Limpiar memoria
                    del asr
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"  [ERROR] en {item['id']}: {str(e)}")
                    continue # Pasar al siguiente sin morir
        
        #guardar depsues de terminar cada modelo
        with open(ruta_resultados, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"--- Guardado: {m_name} listo con {len(data[m_name])} registros ---")

if __name__ == "__main__":
    arreglar()
