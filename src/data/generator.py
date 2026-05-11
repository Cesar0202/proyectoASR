import edge_tts
import asyncio
import os
import json
import random
from disfluency_injector import DisfluencyInjector

class DatasetGenerator:
    # Esta clase sirve para crear los audios de prueba
    def __init__(self, carpeta_salida="results/audios"):
        self.carpeta_salida = carpeta_salida
        # Lista de frases que vmos probar
        self.frases = [
            "La inteligencia artificial cambiará el mundo tal como lo conocemos.",
            "El cambio climático es el mayor desafío de nuestra generación.",
            "La exploración espacial nos permitirá colonizar otros planetas.",
            "La computación cuántica revolucionará la seguridad de las comunicaciones cifradas.",
            "La medicina personalizada utiliza la genética para curar enfermedades.",
            "Las energías renovables son clave para un futuro sostenible.",
            "La nanotecnología permite manipular la materia a escala atómica.",
            "El aprendizaje profundo es una rama avanzada del aprendizaje automático.",
            "La realidad aumentada combina el mundo físico con elementos digitales.",
            "La biotecnología aplicada a la agricultura mejora el rendimiento de los cultivos.",
            "El internet de las cosas conecta objetos cotidianos a la red global.",
            "La ciberseguridad es fundamental para proteger la privacidad de los datos.",
            "Los vehículos autónomos reducirán drásticamente los accidentes de tráfico.",
            "La robótica colaborativa permite que humanos y máquinas trabajen juntos.",
            "El procesamiento de lenguaje natural ayuda a las máquinas a entendernos.",
            "La cadena de bloques garantiza la integridad de las transacciones digitales.",
            "La impresión tres de facilita la fabricación de piezas complejas.",
            "La economía circular busca reducir el desperdicio de recursos.",
            "El aprendizaje colaborativo fomenta el desarrollo de habilidades sociales en el aula.",
            "La innovación tecnológica es el motor del progreso económico."
        ]
        
        #acentos
        self.voces = [
            "es-MX-JorgeNeural", "es-MX-DaliaNeural",
            "es-ES-AlvaroNeural", "es-ES-ElviraNeural",
            "es-CO-GonzaloNeural", "es-AR-ElenaNeural"
        ]
        self.inyector = DisfluencyInjector()
        
        # Crear carpetas
        for tipo in ["repetition", "prolongation", "filler", "block", "mix", "control"]:
            os.makedirs(os.path.join(self.carpeta_salida, tipo), exist_ok=True)

    async def _crear_audio(self, texto, voz, ruta):
        # Llama a edge-tts para guardar el audio
        comunicador = edge_tts.Communicate(texto, voz)
        await comunicador.save(ruta)

    async def ejecutar(self, total=500):
        # Genera los 500 audios
        metadatos = []
        tipos = ["repetition", "prolongation", "filler", "block", "mix"]
        
        for i in range(total):
            frase = random.choice(self.frases)
            tipo = random.choice(tipos)
            sev = random.choice([0.2, 0.5, 0.8])
            inten = random.choice([0.3, 0.7])
            voz = random.choice(self.voces)
            
            # aplciaion del fallo de habla
            nueva_frase = self.inyector.aplicar(frase, tipo, sev, inten)
            
            # Guardar el archivo
            nombre = f"{i}_{tipo}.mp3"
            ruta = os.path.join(self.carpeta_salida, tipo, nombre)
            await self._crear_audio(nueva_frase, voz, ruta)
            
            metadatos.append({
                "id": nombre,
                "speaker_id": voz,
                "disfluency_type": tipo,
                "severity": sev,
                "intensity": inten,
                "text_original": frase,
                "text_modified": nueva_frase,
                "audio_path": ruta
            })
            
        # Guardar la lista de que es cada cosa
        with open("results/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadatos, f, indent=4, ensure_ascii=False)

    async def generar_control(self):
        # Genera los audios sin errores parapoder compararlos
        lista = []
        for i, f in enumerate(self.frases):
            nombre = f"control_{i}.mp3"
            ruta = os.path.join(self.carpeta_salida, "control", nombre)
            voz = random.choice(self.voces)
            await self._crear_audio(f, voz, ruta)
            lista.append({
                "id": nombre, "speaker_id": voz, "disfluency_type": "control",
                "severity": 0.0, "intensity": 0.0, "text_original": f,
                "text_modified": f, "audio_path": ruta
            })
        with open("results/metadata_control.json", "w", encoding="utf-8") as f:
            json.dump(lista, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    g = DatasetGenerator()
    asyncio.run(g.ejecutar(500))
