import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import torch

class ThesaurusMatcher:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.terms_map = {}
        self.variations = []
        self.embeddings = None
        self.load_thesaurus()

    def load_thesaurus(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")

        # Expressão regular melhorada para capturar os blocos de termos
        # Usa um lookahead para dividir nos termos principais
        blocks = re.split(r'\n\s*(?=[A-Z0-9].*?\n\s*Def\.:)', content)
        
        all_phrases = []

        for block in blocks:
            lines = block.strip().split('\n')
            
            if not lines or not lines[0].strip():
                continue
            
            main_term = lines[0].strip()
            self.terms_map[main_term.lower()] = main_term
            
            # Adiciona o termo principal para indexação
            all_phrases.append(main_term.lower())

            # Extrai variações
            for line in lines[1:]:
                if 'Usado por:' in line:
                    variations = re.findall(r'(\b[\w\s]+\b)', line.replace('Usado por:', ''))
                    for variation in variations:
                        variation = variation.strip().lower()
                        if variation:
                            self.terms_map[variation] = main_term
                            all_phrases.append(variation)
                if 'Use:' in line:
                    use_variations = re.findall(r'(\b[\w\s]+\b)', line.replace('Use:', ''))
                    for variation in use_variations:
                        variation = variation.strip().lower()
                        if variation:
                            self.terms_map[variation] = main_term
                            all_phrases.append(variation)

        # Remove duplicatas
        all_phrases = list(set(all_phrases))
        
        print("Carregando modelo semântico...")
        if all_phrases:
            self.embeddings = self.model.encode(all_phrases, convert_to_tensor=True)
            self.variations = all_phrases
            print(f"✅ Thesaurus carregado com {len(self.terms_map)} termos autorizados e {len(all_phrases)} variações.")
        else:
            print("⚠️ Aviso: Nenhuma frase foi extraída do thesaurus. Embeddings não foram criados.")
            self.embeddings = None
            self.variations = []

    def find_best_matches(self, text, threshold=0.5):
        if self.embeddings is None or not text.strip():
            return []

        # Codificar o texto inteiro para uma única representação
        text_embedding = self.model.encode([text], convert_to_tensor=True)
        
        # Calcular a similaridade de cosseno entre o texto e todos os termos
        similarities = cos_sim(text_embedding, self.embeddings)[0]

        # Encontrar os índices dos termos com similaridade acima do limite
        indices = torch.nonzero(similarities > threshold, as_tuple=False).squeeze()
        
        results = set()
        
        if indices.numel() == 0:
            return []
            
        # Garante que 'indices' é uma lista para iteração, mesmo se for apenas um
        if indices.dim() == 0:
            indices = [indices.item()]
        else:
            indices = indices.tolist()

        for idx in indices:
            variation = self.variations[idx]
            authorized_term = self.terms_map.get(variation)
            if authorized_term:
                results.add(authorized_term)

        return sorted(list(results))
