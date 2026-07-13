import streamlit as st
import pandas as pd
import json
import plotly.express as px
import os
import sys

st.set_page_config(page_title="ASR Lab :) - g3", layout="wide", page_icon="🔬")

# diseño para que no se vea el streamlit
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Colores que elegimos para el proyecto */
    :root {
        --accent: #3B82F6;
        --bg-main: #0F172A;
        --bg-card: rgba(30, 41, 59, 0.7);
        --border: rgba(148, 163, 184, 0.1);
    }

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Para que el fondo sea oscuro */
    .stApp {
        background-color: #0F172A;
    }

    /* El titulo de arriba */
    .main-title {
        color: #F8FAFC;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        letter-spacing: -1px;
        margin-bottom: 0px;
    }

    /* Las pestañas de arriba */
    button[data-baseweb="tab"] {
        border-bottom: 2px solid transparent !important;
    }
    button[data-baseweb="tab"] p {
        font-size: 18px !important;
        color: #94A3B8 !important;
        font-weight: 400 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid var(--accent) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="tab-highlight"] {
        display: none !important; /* Quitamos la linea roja por defecto */
    }

    /* Las cajitas de las metricas (los numeros grandes) */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        padding: 1.5rem !important;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-weight: 500;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] div {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }

    /* La barra de la izquierda */
    .stSidebar {
        background-color: #020617 !important;
        border-right: 1px solid var(--border);
    }

    /* Dividers Suaves */
    hr {
        border-top: 1px solid rgba(148, 163, 184, 0.1) !important;
    }

    /* Estilo de los Sliders */
    .stSlider > div > div > div > div {
        background-color: var(--accent) !important;
    }

    </style>
""", unsafe_allow_html=True)

# El titulo principal que sale arriba
st.markdown('<h1 class="main-title">ASR Lab - G3</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: #64748B; margin-top: -5px; font-size: 1rem; font-weight: 400;">Comparativa de que tan bien entienden los modelos cuando hay tartamudez</p>', unsafe_allow_html=True)
st.divider()

# Colores para cada modelo en los graficos
COLOR_MAP = {
    "google": "#636EFA",
    "vosk": "#19D3F3",   
    "wav2vec2": "#00CC96",   
    "whisper_base": "#AB63FA" 
}

# Esto es por si whisper da problemas con el audio
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
except:
    pass

# Para que funcionen los modelos desde cualquier ruta (especialmente en servidores)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src"))
from models.asr_models import ASRManager

# para leer el json de resultados
def cargar_datos():
    ruta = "results/results_final.json"
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Pasar el jnson a una tabla
    filas = []
    for modelo, resultados in data.items():
        for r in resultados:
            # Mapeo de etiquetas según proyecto.txt
            r['severity_label'] = {0.2: "Leve", 0.5: "Moderado", 0.8: "Severo"}.get(r['severity'], "Control")
            r['intensity_label'] = {0.3: "Baja", 0.7: "Alta"}.get(r['intensity'], "Control")
            
            tipo_map = {
                'repetition': "Repeticiones",
                'prolongation': "Prolongaciones",
                'filler': "Fillers (Muletillas)",
                'block': "Bloqueos (Pausas)",
                'mix': "Combinaciones",
                'control': "Control (Limpio)"
            }
            r['disfluency_type'] = tipo_map.get(r['disfluency_type'], r['disfluency_type'])
            filas.append(r)
    return pd.DataFrame(filas)

df = cargar_datos()

st.title("Sistema de Benchmark para Modelos de Voz (ASR)")

if df is None:
    st.error("No hay datos todavía.")
else:
    st.sidebar.title("Controles del Lab")
    st.sidebar.write("Configura los graficos:")
    
    with st.sidebar:
        mod_sel = st.multiselect("Elegir Modelos", df['model'].unique(), default=list(df['model'].unique()))
        
    # Separamos el audio normal de los que tienen disfluencias
    df_control = df[df['intensity_label'] == "Control"]
    df_f = df[df['intensity_label'] != "Control"]

    # Filtramos por lo que eligio el usuario
    df_f = df_f[df_f['model'].isin(mod_sel)]
    df_control = df_control[df_control['model'].isin(mod_sel)]

    # Las 3 pestañas
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Probar Audio", "Analisis de Errores"])

    with tab1:
        # Los numeros de arriba
        st.subheader("Numeros Generales")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        
        wer_control = df_control['wer'].mean() if not df_control.empty else 0
        wer_disfluente = df_f['wer'].mean() if not df_f.empty else 0
        cer_disf = df_f['cer'].mean() if not df_f.empty else 0
        sub_disf = df_f['substitutions'].mean() if not df_f.empty else 0
        del_disf = df_f['deletions'].mean() if not df_f.empty else 0
        ins_disf = df_f['insertions'].mean() if not df_f.empty else 0
        
        m1.metric("WER Control", f"{wer_control:.2%}")
        m2.metric("WER Disfluente", f"{wer_disfluente:.2%}")
        m3.metric("CER Promedio", f"{cer_disf:.2%}")
        m4.metric("Sustituciones", f"{sub_disf:.2f}")
        m5.metric("Eliminaciones", f"{del_disf:.2f}")
        m6.metric("Inserciones", f"{ins_disf:.2f}")

        st.divider()

        # Experimento 1: Base (Control vs Disfluente)
        st.subheader("Experimento 1: Impacto Base de la Disfluencia")
        df_comp = pd.concat([
            df_control.assign(categoria="Control (Limpio)"),
            df_f.assign(categoria="Con Disfluencias")
        ])
        
        res_comp = df_comp.groupby(['categoria', 'model'])['wer'].mean().reset_index()
        fig_base = px.bar(res_comp, x="model", y="wer", color="categoria", barmode="group",
                          text_auto=".2%", title="Promedio de Error: Audio Limpio vs Disfluente",
                          color_discrete_map={"Control (Limpio)": "#1E3A8A", "Con Disfluencias": "#3B82F6"},
                          labels={"model": "Modelo", "wer": "WER", "categoria": "Categoría"})
        
        fig_base.update_layout(template="plotly_dark", yaxis_tickformat=".0%", 
                              xaxis_title="Modelo", yaxis_title="Tasa de Error (WER)")
        st.plotly_chart(fig_base, use_container_width=True)

        st.divider()

        # Análisis Combinado: Severidad e Intensidad
        st.subheader("Análisis Detallado: Evolución del Error (Severidad + Intensidad)")
        st.write("Evolución del error desde el estado base (Control) hasta las combinaciones más extremas de disfluencia.")
        
        def agrupar_sev_int(row):
            if row['severity_label'] == "Control" or row['intensity_label'] == "Control":
                return "0. Base (Control)"
            # Para asegurar el orden correcto en el gráfico
            niveles = {"Leve": "1. Leve", "Moderado": "2. Mod", "Severo": "3. Sev"}
            return f"{niveles.get(row['severity_label'], row['severity_label'])} - {row['intensity_label']}"
            
        df_combined = pd.concat([df_control, df_f])
        df_combined['Combinacion'] = df_combined.apply(agrupar_sev_int, axis=1)
        
        orden_categorias = [
            "0. Base (Control)",
            "1. Leve - Baja", "1. Leve - Alta",
            "2. Mod - Baja", "2. Mod - Alta",
            "3. Sev - Baja", "3. Sev - Alta"
        ]
        
        df_combined['Combinacion'] = pd.Categorical(df_combined['Combinacion'], categories=orden_categorias, ordered=True)
        evolucion = df_combined.groupby(['Combinacion', 'model'])['wer'].mean().reset_index()
        
        fig_evo = px.line(evolucion, x="Combinacion", y="wer", color="model", markers=True,
                          title="Degradación Progresiva del WER (Base vs Combos)",
                          color_discrete_map=COLOR_MAP,
                          labels={"Combinacion": "Escenario (Severidad - Intensidad)", "wer": "WER Promedio", "model": "Modelo"})
        fig_evo.update_layout(template="plotly_dark", yaxis_tickformat=".0%", xaxis_tickangle=-45)
        st.plotly_chart(fig_evo, use_container_width=True)

              # Experimento 2: Tipos de Disfluencia
        st.subheader("Experimento 2: Análisis por Categoría")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Error promedio por tipo**")
            f1 = px.bar(df_f.groupby(['disfluency_type', 'model'])['wer'].mean().reset_index(), 
                       x='disfluency_type', y='wer', color='model', barmode='group',
                       color_discrete_map=COLOR_MAP)
            f1.update_layout(template="plotly_dark")
            st.plotly_chart(f1, use_container_width=True)
            
        with c2:
            st.write("**Mapa de Sensibilidad (Radar)**")
            radar = df_f.groupby(['model', 'disfluency_type'])['wer'].mean().reset_index()
            f3 = px.line_polar(radar, r='wer', theta='disfluency_type', color='model', line_close=True,
                               color_discrete_map=COLOR_MAP)
            f3.update_layout(template="plotly_dark")
            st.plotly_chart(f3, use_container_width=True)

        st.divider()

        # Experimento 3: Hablantes
        st.subheader("Experimento 3: Diferentes Voces")nto 5: Diferentes Voces")
        speak_df = df_f.groupby(['speaker_id', 'model'])['wer'].mean().reset_index()
        f6 = px.bar(speak_df, x='speaker_id', y='wer', color='model', barmode='group',
                   color_discrete_map=COLOR_MAP,
                   title="¿A qué modelos les cuesta más ciertas voces?")
        f6.update_layout(template="plotly_dark", xaxis_tickangle=-45)
        st.plotly_chart(f6, use_container_width=True)

    with tab2:
        st.header("Probar un audio")
        modo = st.radio("¿Qué quieres hacer?", ["Subir mi audio", "Usar uno del sistema"])
        
        audio = None
        txt_ref = ""
        
        if modo == "Subir mi audio":
            subido = st.file_uploader("Audio", type=["wav", "mp3"])
            txt_ref = st.text_input("Lo que dice el audio")
            if subido:
                audio = f"temp/{subido.name}"
                os.makedirs("temp", exist_ok=True)
                with open(audio, "wb") as f: f.write(subido.getbuffer())
                st.audio(audio)
        else:
            sel = st.selectbox("Elegir audio", df['audio_path'].unique())
            if sel:
                # Arreglar barras para Linux/Windows y ver si el archivo de verdad existe
                audio_path_fixed = sel.replace("\\", "/")
                if os.path.exists(audio_path_fixed):
                    audio = audio_path_fixed
                    fila = df[df['audio_path'] == sel].iloc[0]
                    txt_ref = fila['text_original']
                    st.audio(audio)
                    st.write(f"Texto: {txt_ref}")
                else:
                    st.warning(f"El archivo de audio no esta en el servidor: {audio_path_fixed}")
                    st.info("Puedes subir tu propio audio en la opcion de arriba para probar.")
            
        if audio:
            m = st.selectbox("Modelo", ["whisper_base", "google", "vosk", "wav2vec2"])
            if st.button("Transcribir ahora"):
                with st.spinner("Procesando audio..."):
                    import jiwer
                    asr = ASRManager(model_type=m)
                    pred = asr.transcribir(audio)
                    st.success(f"Dijo: {pred}")
                    
                    if txt_ref:
                        # Limpiamos el texto para que tildes y comas no cuenten como error
                        import re, unicodedata
                        def normalizar(t):
                            if not t: return ""
                            t = t.lower()
                            t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
                            t = re.sub(r'[^\w\s]', '', t)
                            return t.strip()

                        ref_norm = normalizar(txt_ref)
                        pred_norm = normalizar(pred)

                        calc = jiwer.process_words(ref_norm, pred_norm)
                        cer_val = jiwer.cer(ref_norm, pred_norm)
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("WER", f"{calc.wer:.2%}")
                        m2.metric("CER", f"{cer_val:.2%}")
                        m3.metric("Sustituciones", calc.substitutions)
                        m4.metric("Eliminaciones", calc.deletions)
                        m5.metric("Inserciones", calc.insertions)
                        
                        # Listamos los errores justo debajo sin tanto espacio
                        ref_words = ref_norm.split()
                        hyp_words = pred_norm.split()
                        
                        st.markdown("<h4 style='margin-top: 25px; margin-bottom: 10px;'>Errores encontrados:</h4>", unsafe_allow_html=True)
                        for chunk in calc.alignments[0]:
                            if chunk.type == 'substitute':
                                for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx), 
                                               range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                                    st.write(f"Sustitución: '{ref_words[i]}' por '{hyp_words[j]}'")
                            elif chunk.type == 'delete':
                                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                                    st.write(f"Eliminación: Se perdió la palabra '{ref_words[i]}'")
                            elif chunk.type == 'insert':
                                for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                                    st.write(f"Inserción: Se añadió la palabra '{hyp_words[j]}'")

    with tab3:
        st.header("Que aprendimos de los datos")
        # 1. ¿Borraron las palabras o las intentaron escribir?
        st.subheader("1. ¿Como se portan con las repeticiones?")
        df_rep = df[df['disfluency_type'] == "Repeticiones"]
        if not df_rep.empty:
            rep_stats = df_rep.groupby('model').agg({'deletions': 'mean', 'substitutions': 'mean', 'insertions': 'mean'}).reset_index()
            # Les ponemos nombres claros
            rep_stats = rep_stats.rename(columns={
                'deletions': 'Las borro', 
                'substitutions': 'Las cambio', 
                'insertions': 'Invento palabras'
            })
            
            st.write("Aca vemos si los modelos borran los tartamudeos o intentan ponerlos en el texto")
            f_rep = px.bar(rep_stats, x='model', y=['Las borro', 'Las cambio', 'Invento palabras'], 
                          title="Distribución de Errores en Audios con Repeticiones",
                          color_discrete_sequence=["#1E3A8A", "#3B82F6", "#94A3B8"], 
                          labels={"value": "Promedio de palabras", "variable": "Tipo de Error", "model": "Modelo"})
            f_rep.update_layout(template="plotly_dark")
            st.plotly_chart(f_rep, use_container_width=True)
            st.info("Nota: Una tasa alta de 'Eliminaciones' indica que el modelo está filtrando la tartamudez automáticamente.")

        st.divider()

        # 2análisis de Robustez por Tipo
        st.subheader("2. Comparación de Robustez: ¿Quién falla en qué?")
        # Encontrar el mejor modelo por cada tipo
        best_models = []
        for t in df['disfluency_type'].unique():
            if t == "Control (Limpio)": continue
            df_t = df[df['disfluency_type'] == t]
            if not df_t.empty:
                best = df_t.groupby('model')['wer'].mean().idxmin()
                val = df_t.groupby('model')['wer'].mean().min()
                best_models.append({"Tipo": t, "Mejor Modelo": best, "WER Mínimo": val})
        
        st.table(pd.DataFrame(best_models).style.format({"WER Mínimo": "{:.2%}"}))

        st.divider()

        # 3- Ejemplos para Análisis Semántico
        st.subheader("3. Muestras para Análisis Semántico")
        st.write("A continuación se presentan ejemplos reales del benchmark con error alto para evaluar si se mantiene el significado.")
        
        # Seleccionar 5 ejemplos con WER alto
        muestras = df_f[df_f['wer'] > 0.4].sample(min(5, len(df_f[df_f['wer'] > 0.4]))) if not df_f[df_f['wer'] > 0.4].empty else pd.DataFrame()
        if not muestras.empty:
            for _, m in muestras.iterrows():
                with st.expander(f"Caso: {m['disfluency_type']} (WER: {m['wer']:.2%}) - Modelo: {m['model']}"):
                    st.write(f"**Referencia:** {m['text_original']}")
                    st.write(f"**Transcripción:** {m['prediction']}")
                    st.write(f"*Análisis:* ¿Se entiende el mensaje principal?")
        else:
            st.write("No hay ejemplos con error suficiente para este análisis en la selección actual.")

        st.divider()

        # 4. Conclusiones (creo se se borrará)
        st.subheader("4. Hallazgos finales")
        c_fin1, c_fin2 = st.columns(2)
        
        with c_fin1:
            st.markdown("""
            **Sobre los errores:**
            * **Borrar palabras:** Google y Vosk suelen borrar las repeticiones para que la frase se lea mas limpia. Whisper intenta poner todo lo que escucha, lo que a veces sube el error pero es mas real.
            * **Inventar palabras:** Cuando no entienden bien la disfluencia, Whisper o Wav2Vec2 a veces ponen palabras que nada que ver (alucinaciones).
            * **Contexto:** Casi siempre las palabras que estan al lado de la falla se entienden bien.
            """)
        
        with c_fin2:
            st.markdown("""
            **Cual es mas fuerte:**
            * **Sentido de la frase:** Casi siempre se entiende que quiso decir el audio, aunque el WER sea alto. El fallo es mas de forma que de significado.
            * **Ranking:** 
                - **Vosk:** Es muy bueno cuando hay pocos errores.
                - **Whisper:** Es el mejor manteniendo el mensaje original.
                - **Wav2Vec2:** Le cuesta mucho cuando el tartamudeo es largo.
            """)
        
        st.success("Análisis finalizado: La robustez de un ASR debe medirse por su capacidad de mantener el mensaje, no solo por su precisión ortográfica ante disfluencias.")