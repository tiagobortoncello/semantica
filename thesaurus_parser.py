# thesaurus_parser.py
import re
from sentence_transformers import SentenceTransformer
import numpy as np

class ThesaurusMatcher:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.terms = []  # Lista de termos padrão (ex: "Peculato")
        self.variations = []  # Todas as variações (Use:, Usado por:)
        self.embeddings = None
        self.load_thesaurus()

    def load_thesaurus(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")

        # Limpeza básica
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\s+', ' ', content)
        blocks = content.split('\nSituação: Ativo')
        blocks = [b.strip() for b in blocks if b.strip()]

        all_phrases = []
        self.terms = []

        for block in blocks:
            lines = block.strip().split('\n')
            term = None
            use = None
            usado_por = []

            for line in lines:
                line = line.strip()
                if not term and ':' not in line:
                    term = line
                elif line.startswith('Use:'):
                    use = line[len('Use:'):].strip()
                elif line.startswith('Usado por:'):
                    sin_list = line[len('Usado por:'):].strip()
                    usado_por = [s.strip() for s in sin_list.split(',')]

            if not term:
                continue

            termo_padrao = use or term
            self.terms.append(termo_padrao)

            # Adiciona todas as variações
            variations = [term]
            if use:
                variations.append(use)
            variations.extend(usado_por)

            all_phrases.extend([v.lower() for v in variations])

        # Gerar embeddings
        print("Carregando modelo semântico...")
        self.embeddings = self.model.encode(all_phrases, convert_to_tensor=True)
        self.variations = all_phrases
        print(f"✅ Thesaurus carregado com {len(self.terms)} termos e {len(all_phrases)} variações.")

    def find_best_matches(self, text, threshold=0.6):
        from sentence_transformers.util import cos_sim

        # Extrair frases do texto
        phrases = re.findall(r'\b[\w\s]+\b', text.lower())
        phrases = [p.strip() for p in phrases if len(p.strip().split()) >= 2]

        results = set()

        for phrase in phrases:
            if len(phrase) < 5:
                continue

            phrase_emb = self.model.encode([phrase], convert_to_tensor=True)
            sims = cos_sim(phrase_emb, self.embeddings)[0]
            best_idx = sims.argmax().item()
            best_score = sims[best_idx].item()

            if best_score > threshold:
                variation = self.variations[best_idx]
                termo_padrao = self.terms[self.variations[best_idx].lower() == np.array(self.variations).lower()]
                if len(termo_padrao) > 0:
                    results.add(termo_padrao[0])

        return list(results)
