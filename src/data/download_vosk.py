import urllib.request
import zipfile
import os

def download_vosk():
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-es-0.22.zip"
    dest_path = "models/vosk-model-small-es.zip"
    model_dir = "models"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if not os.path.exists("models/vosk-model-es"):
        print("Descargando modelo Vosk pequeño (aprox 40MB)...")
        urllib.request.urlretrieve(model_url, dest_path)
        
        print("Descomprimiendo...")
        with zipfile.ZipFile(dest_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
            
        # Rename to the standard path we want
        extracted_folder = os.path.join(model_dir, "vosk-model-small-es-0.22")
        os.rename(extracted_folder, os.path.join(model_dir, "vosk-model-es"))
        os.remove(dest_path)
        print("¡Vosk listo!")

if __name__ == "__main__":
    download_vosk()
