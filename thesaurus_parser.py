import re
from sentence_transformers import SentenceTransformer
import numpy as np

class ThesaurusMatcher:
    def __init__(self, file_path):
        self.file_path = file_path
        # Use um modelo de linguagem otimizado para tarefas de similaridade
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

        # Limpeza básica do conteúdo
        # Substitui múltiplos espaços e quebras de linha por um único espaço
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Expressão regular para encontrar blocos de termos
        # Captura o termo principal, o "Use:" e as frases de "Usado por:"
        block_pattern = re.compile(
            r'^(.*?)\s*Def\.:.*?\s*Use:\s*(.*?)\s+Situação: Ativo\s*|'
            r'^(.*?)\s*Def\.:.*?\s*Usado por:\s*(.*?)\s+Situação: Ativo\s*|'
            r'^(.*?)\s*Def\.:.*?\s+Situação: Ativo\s*',
            re.DOTALL
        )

        all_phrases = []
        self.terms = []

        # Analisa o conteúdo em blocos e extrai os termos
        blocks = re.split(r'\n(?=\S)', content) # Divide por quebra de linha seguida por caractere não-espaço

        for block in blocks:
            lines = block.strip().split('\n')
            
            # Pega o primeiro termo, que é sempre o principal
            if not lines:
                continue
            
            main_term = lines[0].strip()
            self.terms.append(main_term)
            
            # Adiciona o termo principal e variações à lista de frases
            all_phrases.append(main_term)
            for line in lines:
                # Extrai frases de "Use:"
                if 'Use:' in line:
                    use_term = line.replace('Use:', '').strip()
                    all_phrases.append(use_term)
                # Extrai frases de "Usado por:"
                if 'Usado por:' in line:
                    # Captura todas as variações em uma única linha, se houver
                    usado_por_terms = re.findall(r'(\b[\w\s]+\b)', line.replace('Usado por:', ''))
                    all_phrases.extend([term.strip() for term in usado_por_terms if term.strip()])

        # Remove duplicatas e espaços extras
        all_phrases = list(set([phrase.strip().lower() for phrase in all_phrases if phrase.strip()]))
        
        # Gerar embeddings
        print("Carregando modelo semântico...")
        if all_phrases:
            self.embeddings = self.model.encode(all_phrases, convert_to_tensor=True)
            self.variations = all_phrases
            print(f"✅ Thesaurus carregado com {len(self.terms)} termos e {len(all_phrases)} variações.")
        else:
            print("⚠️ Aviso: Nenhuma frase foi extraída do thesaurus. Embeddings não foram criados.")
            self.embeddings = None
            self.variations = []

    def find_best_matches(self, text, threshold=0.6):
        from sentence_transformers.util import cos_sim

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
                variation = self.variations[best_idx]
                results.add(variation)

        return list(results)
