# streamlit_app.py
import streamlit as st
from thesaurus_parser import ThesaurusMatcher

st.title("🔍 Indexador Semântico com Thesaurus")
st.markdown("""
Cole um texto. O sistema vai sugerir **termos padronizados** com base no significado, não só palavras exatas.
""")

# Carregar o matcher
@st.cache_resource
def carregar_thesaurus():
    return ThesaurusMatcher('sth.txt')

try:
    matcher = carregar_thesaurus()
except Exception as e:
    st.error(f"Erro: {e}")
    st.stop()

texto = st.text_area("Cole seu texto aqui:", height=200)

if st.button("🔍 Sugerir Termos (Semântico)"):
    if not texto.strip():
        st.warning("Por favor, cole um texto.")
    else:
        with st.spinner("Analisando significado..."):
            termos = matcher.find_best_matches(texto, threshold=0.5)
        if termos:
            st.success("✅ Termos sugeridos:")
            for t in termos:
                st.markdown(f"- **{t}**")
        else:
            st.info("❌ Nenhum termo semântico encontrado. Tente um texto mais claro.")
