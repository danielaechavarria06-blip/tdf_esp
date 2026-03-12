import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer
import json
from streamlit_lottie import st_lottie

# ---------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------

st.set_page_config(
    page_title="Demo TF-IDF",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------
# CARGAR ANIMACIÓN LOTTIE
# ---------------------------

def load_lottiefile(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# ⚠️ CAMBIA AQUÍ EL NOMBRE DE TU JSON
lottie_animation = load_lottiefile("animacion.json")

# ---------------------------
# TÍTULO
# ---------------------------

st.title("🔍 Demo TF-IDF en Español")

# animación centrada
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st_lottie(
        lottie_animation,
        speed=1,
        height=250,
        key="initial_animation"
    )

# ---------------------------
# DOCUMENTOS DE EJEMPLO
# ---------------------------

default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

# ---------------------------
# GRÁFICA SIMPLE (SIN MATPLOTLIB)
# ---------------------------

st.subheader("📊 Palabras por documento")

documents_chart = default_docs.split("\n")
doc_lengths = [len(d.split()) for d in documents_chart]

chart_df = pd.DataFrame({
    "Documento": [f"Doc {i+1}" for i in range(len(doc_lengths))],
    "Palabras": doc_lengths
})

st.bar_chart(chart_df.set_index("Documento"))

# ---------------------------
# STEMMER
# ---------------------------

stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):

    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)

    tokens = [t for t in text.split() if len(t) > 1]

    stems = [stemmer.stem(t) for t in tokens]

    return stems

# ---------------------------
# LAYOUT PRINCIPAL
# ---------------------------

col1, col2 = st.columns([2, 1])

with col1:

    text_input = st.text_area(
        "📝 Documentos (uno por línea):",
        default_docs,
        height=150
    )

    question = st.text_input(
        "❓ Escribe tu pregunta:",
        "¿Dónde juegan el perro y el gato?"
    )

with col2:

    st.markdown("### 💡 Preguntas sugeridas:")

    if st.button("¿Dónde juegan el perro y el gato?", use_container_width=True):
        st.session_state.question = "¿Dónde juegan el perro y el gato?"
        st.rerun()

    if st.button("¿Qué hacen los niños en el parque?", use_container_width=True):
        st.session_state.question = "¿Qué hacen los niños en el parque?"
        st.rerun()

    if st.button("¿Cuándo cantan los pájaros?", use_container_width=True):
        st.session_state.question = "¿Cuándo cantan los pájaros?"
        st.rerun()

    if st.button("¿Dónde suena la música alta?", use_container_width=True):
        st.session_state.question = "¿Dónde suena la música alta?"
        st.rerun()

    if st.button("¿Qué animal maúlla durante la noche?", use_container_width=True):
        st.session_state.question = "¿Qué animal maúlla durante la noche?"
        st.rerun()

# ---------------------------
# ACTUALIZAR PREGUNTA
# ---------------------------

if 'question' in st.session_state:
    question = st.session_state.question

# ---------------------------
# BOTÓN ANALIZAR
# ---------------------------

if st.button("🔍 Analizar", type="primary"):

    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")

    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")

    else:

        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1
        )

        X = vectorizer.fit_transform(documents)

        st.markdown("### 📊 Matriz TF-IDF")

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        st.dataframe(df_tfidf.round(3), use_container_width=True)

        # similitud

        question_vec = vectorizer.transform([question])

        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()

        best_doc = documents[best_idx]

        best_score = similarities[best_idx]

        st.markdown("### 🎯 Respuesta")

        st.markdown(f"**Tu pregunta:** {question}")

        if best_score > 0.01:

            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")

        else:

            st.warning(f"**Respuesta (baja confianza):** {best_doc}")
            st.info(f"📉 Similitud: {best_score:.3f}")
