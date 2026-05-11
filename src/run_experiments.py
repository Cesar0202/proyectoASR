import json
import pandas as pd
import os

def run_analysis(results_path="results/results.json"):
    if not os.path.exists(results_path):
        print("Error: No se encontró el archivo de resultados.")
        return
        
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # Reporte 1: Promedio de WER y CER por Severidad
    severity_report = df.groupby("severity")[["wer", "cer"]].mean().reset_index()
    severity_report["severity_label"] = severity_report["severity"].map({0.2: "Leve", 0.5: "Moderado", 0.8: "Severo"})
    
    print("\n--- Reporte por Severidad ---")
    print(severity_report[["severity_label", "wer", "cer"]])
    
    #Reporte 2: Comparacion de texto original/predicción
    print("\n--- Ejemplos de Transcripción ---")
    for _, row in df.head(5).iterrows():
        print(f"\nID: {row['id']}")
        print(f"Severity: {row['severity']}")
        print(f"Modificado (Ground Truth): {row['text_modified']}")
        print(f"Predicción ASR: {row['prediction']}")
        print(f"WER: {row['wer']:.4f}")

    # Guardar reporte
    os.makedirs("results/reports", exist_ok=True)
    severity_report.to_csv("results/reports/severity_analysis.csv", index=False)
    
    # Resumen para el usuario
    with open("results/reports/summary.md", "w", encoding="utf-8") as f:
        f.write("# Resumen de Experimentos\n\n")
        f.write("## Análisis por Severidad\n")
        f.write(severity_report[["severity_label", "wer", "cer"]].to_markdown(index=False))
        f.write("\n\n## Conclusiones Preliminares\n")
        f.write("- La severidad impacta directamente en el WER.\n")
        f.write("- Los modelos ASR tienden a 'limpiar' o 'ignorar' ciertas disfluencias.\n")

if __name__ == "__main__":
    run_analysis()
