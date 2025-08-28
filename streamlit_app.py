import streamlit as st
from thesaurus_parser import parse_sth_file
import re
from sentence_transformers import SentenceTransformer, util

# --- Configuração do modelo semântico ---
@st.cache_resource
def carregar_modelo():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- Carregar thesaurus ---
@st.cache_data
def carregar_thesaurus():
    return parse_sth_file('sth.txt')

st.title("🔍 Indexador Automático com Thesaurus + Semântica")

st.markdown("""
Cole um texto e escolha o tipo de documento (**Requerimento, Proposição ou Norma**).  
O sistema identifica palavras e sugere **termos padronizados** do thesaurus.
""")

tipo = st.radio("📌 Escolha o tipo de documento:", ["Requerimento", "Proposição", "Norma"])

try:
    thesaurus, word_map = carregar_thesaurus()
    st.success(f"✅ Thesaurus carregado! {len(word_map)} variações mapeadas.")
except Exception as e:
    st.error(f"❌ Erro ao carregar o arquivo 'sth.txt': {e}")
    st.stop()

texto = st.text_area("Cole seu texto aqui:", height=200, placeholder="Ex: O servidor fez um acordo judicial e pagou imposto atrasado...")

if st.button("🔍 Sugerir Termos"):
    if not texto.strip():
        st.warning("Por favor, cole um texto para análise.")
    else:
        palavras = re.findall(r'\b[a-zA-ZÀ-ÿçÇãÃõÕ]+\b', texto.lower())

        # --- Matching exato ---
        termos_encontrados = set()
        detalhes_exatos = []
        for palavra in palavras:
            if palavra in word_map:
                termo = word_map[palavra]
                if termo not in termos_encontrados:
                    termos_encontrados.add(termo)
                    detalhes_exatos.append(f"🔹 `{palavra}` → **{termo}**")

        # --- Matching semântico ---
        modelo = carregar_modelo()
        embeddings_termos = {t: modelo.encode(t, convert_to_tensor=True) for t in thesaurus.keys()}
        embedding_texto = modelo.encode(texto, convert_to_tensor=True)

        similares = []
        for termo, emb in embeddings_termos.items():
            score = float(util.cos_sim(embedding_texto, emb))
            if score > 0.70:  # limiar de similaridade
                similares.append((termo, score))

        similares = sorted(similares, key=lambda x: x[1], reverse=True)

        # --- Exibir resultados ---
        if detalhes_exatos:
            st.subheader("✅ Termos encontrados (exatos)")
            st.markdown("\n".join(detalhes_exatos))

        if similares:
            st.subheader("🤖 Sugestões semânticas")
            for termo, score in similares[:10]:
                st.markdown(f"🔸 **{termo}** (similaridade: {score:.2f})")

        if not detalhes_exatos and not similares:
            st.info("❌ Nenhum termo encontrado.")

        st.markdown("---")
        st.markdown(f"📌 Contexto selecionado: **{tipo}**")
