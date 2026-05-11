import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))
from evaluator import Evaluator

evaluator = Evaluator(metadata_path="results/metadata_control.json")
evaluator.correr_evaluacion(modelos=['google', 'vosk', 'wav2vec2', 'whisper_base'])
print("Evaluación de control completada.")
