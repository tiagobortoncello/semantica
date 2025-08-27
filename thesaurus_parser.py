import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np

class ThesaurusMatcher:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.terms_map = {}  # Dicionário para mapear variações para termos autorizados
        self.variations = []
        self.embeddings = None
        self.load_thesaurus()

    def load_thesaurus(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")

        # O padrão de quebra de linha com espaços é a chave para a análise
        blocks = re.split(r'\n\s*\n', content)
        
        all_phrases = []

        for block in blocks:
            lines = block.strip().split('\n')
            
            # Pega o primeiro termo, que é o principal
            if not lines:
                continue
            
            main_term = lines[0].strip()
            self.terms_map[main_term.lower()] = main_term

            # Processa o resto do bloco para encontrar "Use:" e "Usado por:"
            variations_in_block = []
            for line in lines[1:]:
                if 'Use:' in line:
                    variations = re.findall(r'(\b[\w\s]+\b)', line.replace('Use:', ''))
                    variations_in_block.extend([v.strip() for v in variations])
                elif 'Usado por:' in line:
                    variations = re.findall(r'(\b[\w\s]+\b)', line.replace('Usado por:', ''))
                    variations_in_block.extend([v.strip() for v in variations])
            
            # Mapeia todas as variações do bloco para o termo principal
            for variation in variations_in_block:
                variation_lower = variation.lower()
                self.terms_map[variation_lower] = main_term
                all_phrases.append(variation_lower)

            # Adiciona o termo principal também como uma frase para ser indexada
            all_phrases.append(main_term.lower())

        # Remove duplicatas
        all_phrases = list(set(all_phrases))

        # Gerar embeddings
        print("Carregando modelo semântico...")
        if all_phrases:
            self.embeddings = self.model.encode(all_phrases, convert_to_tensor=True)
            self.variations = all_phrases
            print(f"✅ Thesaurus carregado com {len(self.terms_map)} termos autorizados e {len(all_phrases)} variações.")
        else:
            print("⚠️ Aviso: Nenhuma frase foi extraída do thesaurus. Embeddings não foram criados.")
            self.embeddings = None
            self.variations = []

    def find_best_matches(self, text, threshold=0.6):
        if self.embeddings is None:
            return []

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
                # Usa o mapeamento para obter o termo autorizado
                matched_variation = self.variations[best_idx]
                authorized_term = self.terms_map.get(matched_variation, None)
                if authorized_term:
                    results.add(authorized_term)

        return list(results)
