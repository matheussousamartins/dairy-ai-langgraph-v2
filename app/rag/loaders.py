"""
rag/loaders.py — Estratégias de chunking para documentos

Este módulo divide textos grandes em pedaços menores (chunks) para
armazenar no vector store. É uma ADAPTAÇÃO DIRETA do app/rag/loaders.py
do projeto original do curso, com as seguintes mudanças:

Original (curso):
  - 3 estratégias: fixed, markdown, semantic
  - Tamanhos fixos no código (chunk_size=800, overlap=200)
  - Lê apenas arquivos .md de um diretório

Adaptado (laticínios):
  - Mesmas 3 estratégias (código quase idêntico)
  - Tamanhos vêm do config.py (variam por tipo de documento)
  - Aceita texto direto (não precisa ser arquivo)
  - Adiciona separadores específicos para legislação ("Art. ", "§ ")

Por que chunking importa?
O LLM tem limite de contexto (ex: 128K tokens para GPT-4o). Se enviar
um PDF inteiro de 200 páginas, não cabe. Além disso, embeddings funcionam
melhor com textos curtos e focados — um chunk de 800 caracteres sobre
"filagem de mussarela" gera um vetor mais preciso do que um capítulo
inteiro sobre "fabricação de queijos".

O tamanho ideal do chunk depende do tipo de documento:
  - Legislação (600 chars): artigos são curtos e independentes.
    Chunk grande mistura artigos diferentes → busca imprecisa.
  - Manuais técnicos (1200 chars): seções são longas e interconectadas.
    Chunk pequeno corta no meio de um procedimento → contexto incompleto.
  - FAQ (500 chars): perguntas e respostas são curtas.
    Chunk grande mistura múltiplas Q&As → confunde o LLM.

Overlap (sobreposição):
Quando cortamos o texto em chunks, o final de um chunk se sobrepõe
com o início do próximo. Isso garante que informações na fronteira
entre dois chunks não se percam.
  Exemplo com overlap=100:
    Chunk 1: "...a temperatura de filagem deve ser 80°C para"
    Chunk 2: "deve ser 80°C para garantir a elasticidade da massa."
  Sem overlap, "80°C" poderia ficar no final do chunk 1 e "elasticidade"
  no início do chunk 2, e nenhum dos dois teria a frase completa.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# ============================================================
# Importação condicional do SemanticChunker
# ============================================================
# O SemanticChunker é um splitter avançado que usa embeddings para
# decidir onde cortar o texto. Em vez de cortar por tamanho fixo,
# ele detecta mudanças de assunto (quando o embedding de um parágrafo
# é muito diferente do seguinte, é um bom ponto de corte).
#
# O problema é que esse classe existe em pacotes diferentes dependendo
# da versão instalada:
#   - langchain-text-splitters >= 0.3.9 → importa de lá
#   - langchain-experimental >= 0.3.4 → importa de lá
#   - Nenhum dos dois instalado → não disponível
#
# O try/except tenta ambos e define _HAS_SEMANTIC = True se encontrar.
# Se não encontrar, a estratégia "semantic" levanta erro explícito
# em vez de falhar com ImportError críptico.
#
# Este bloco é IDÊNTICO ao do projeto original (loaders.py linhas 17-31).

_HAS_SEMANTIC = False
SemanticChunker = None

try:
    from langchain_text_splitters import SemanticChunker as _SC
    SemanticChunker = _SC
    _HAS_SEMANTIC = True
except Exception:
    try:
        from langchain_experimental.text_splitter import SemanticChunker as _SC
        SemanticChunker = _SC
        _HAS_SEMANTIC = True
    except Exception:
        SemanticChunker = None
        _HAS_SEMANTIC = False


# ============================================================
# Estratégia 1: Fixed (corte por tamanho)
# ============================================================

def split_fixed(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[str]:
    """Divide texto em chunks de tamanho fixo com overlap.
    
    Esta é a estratégia mais simples e robusta. Funciona bem para
    qualquer tipo de documento. É a mesma do projeto original.
    
    O RecursiveCharacterTextSplitter tenta cortar em pontos "naturais":
    primeiro por "\n\n" (parágrafos), depois por "\n" (linhas),
    depois por ". " (frases), e por último por " " (palavras).
    Se nenhum separador couber no chunk_size, corta por caractere.
    
    Parâmetros:
        text: Texto completo a ser dividido.
        chunk_size: Tamanho máximo de cada chunk em caracteres.
        chunk_overlap: Quantos caracteres de sobreposição entre chunks.
    
    Retorna:
        Lista de strings, cada uma é um chunk.
    
    Exemplo com chunk_size=20, overlap=5:
        Input:  "A mussarela é um queijo de massa filada muito popular."
        Output: ["A mussarela é um", "um queijo de massa", "massa filada muito", "muito popular."]
        Note que "um", "massa", "muito" aparecem em dois chunks (overlap).
    """
    # Separadores customizados para documentos de laticínios.
    # A ordem importa: o splitter tenta o primeiro, se não couber
    # no chunk_size, tenta o segundo, e assim por diante.
    #
    # Diferença do original: adicionamos "\nArt. " e "\n§ " para
    # legislação (corta antes de artigos e parágrafos legais).
    separators = [
        "\n## ",     # Cabeçalho Markdown nível 2
        "\n### ",    # Cabeçalho Markdown nível 3
        "\n\n",      # Parágrafo (dupla quebra de linha)
        "\nArt. ",   # Artigo de legislação (ex: "Art. 15.")
        "\n§ ",      # Parágrafo de legislação (ex: "§ 1º")
        "\n",        # Linha simples
        ". ",        # Fim de frase
        " ",         # Espaço entre palavras (último recurso)
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,          # Mede tamanho por caracteres
        separators=separators,
    )
    
    return splitter.split_text(text)


# ============================================================
# Estratégia 2: Markdown (corte por cabeçalhos)
# ============================================================

def split_markdown(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    max_section_ratio: float = 1.5,
    max_section_chars: Optional[int] = None,
) -> List[str]:
    """Divide texto usando a estrutura de cabeçalhos Markdown.
    
    Ideal para documentos que passaram pela conversão LLM → Markdown
    no pipeline de ingestão. Os cabeçalhos (#, ##, ###) definem
    seções lógicas, e cada seção vira um chunk.
    
    O MarkdownHeaderTextSplitter reconhece:
      # Título    → seção de nível 1
      ## Seção    → seção de nível 2
      ### Subseção → seção de nível 3
    
    Cada seção vira um chunk que inclui o caminho de cabeçalhos
    como contexto. Exemplo:
      # Mussarela
      ## Filagem
      A filagem deve ser feita a 80°C...
    
    O chunk resultante seria:
      "Mussarela > Filagem\nA filagem deve ser feita a 80°C..."
    
    Isso é melhor que corte fixo porque o chunk sabe "de onde veio"
    na estrutura do documento.
    
    Se uma seção for maior que chunk_size, ela é mantida inteira
    (não é subdividida). Para documentos com seções muito grandes,
    use split_fixed como fallback.
    
    Código base idêntico ao original (loaders.py linhas 53-62),
    com adição de fallback para seções grandes.
    """
    # Define quais cabeçalhos usar como pontos de corte
    # e como identificá-los nos metadados
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),      # Título principal
            ("##", "h2"),     # Seção
            ("###", "h3"),    # Subseção
            ("####", "h4"),   # Sub-subseção (ex: PRINCÍPIO, REAGENTES, PROCEDIMENTO por método)
        ]
    )

    # split_text retorna objetos Document do LangChain,
    # cada um com .page_content (texto) e .metadata (cabeçalhos)
    docs = splitter.split_text(text)

    # Monta cada chunk com o caminho de cabeçalhos como prefixo
    chunks = []
    for doc in docs:
        # Extrai os cabeçalhos do metadata
        # Ex: {"h1": "IN 68", "h2": "ACIDEZ", "h3": "Método A", "h4": "REAGENTES"}
        #     → "IN 68 > ACIDEZ > Método A > REAGENTES"
        headers = []
        for level in ["h1", "h2", "h3", "h4"]:
            if level in doc.metadata:
                headers.append(doc.metadata[level])
        
        # Monta o chunk: cabeçalhos + conteúdo
        prefix = " > ".join(headers)
        content = doc.page_content
        
        if prefix:
            chunk_text = f"{prefix}\n{content}"
        else:
            chunk_text = content
        
        # Se o chunk é maior que o limite configurado, subdivide com split_fixed
        # Isso é uma melhoria em relação ao original, que retornava
        # seções grandes inteiras (podendo exceder o limite de tokens)
        split_threshold = int(chunk_size * max_section_ratio)
        if max_section_chars is not None:
            split_threshold = min(split_threshold, int(max_section_chars))

        if len(chunk_text) > split_threshold:
            sub_chunks = split_fixed(chunk_text, chunk_size, chunk_overlap)
            chunks.extend(sub_chunks)
        else:
            chunks.append(chunk_text)

    # Pós-processamento conservador: evita chunk "somente título".
    # Mantém a qualidade já validada, removendo ruído de recuperação.
    def _is_heading_only_chunk(chunk: str) -> bool:
        text = (chunk or "").strip()
        if not text:
            return False
        if "\n" in text:
            head, tail = text.split("\n", 1)
            if tail.strip():
                return False
            text = head.strip()

        words = re.findall(r"\b[\wÀ-ÿ]{2,}\b", text)
        if len(text) > 220 or len(words) > 24:
            return False

        # Heurísticas para heading/path de legislação/markdown.
        if " > " in text:
            return True
        if re.match(r"^(Art\.\s*\d+[A-Za-z]?|§\s*\d+|[#]{1,3}\s+)", text, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(INSTRUÇÃO NORMATIVA|INSTRUC[AÃ]O NORMATIVA)\b", text, flags=re.IGNORECASE):
            return True
        return False

    def _merge_heading_only_chunks(items: List[str]) -> List[str]:
        if not items:
            return items
        merged: List[str] = []
        i = 0
        while i < len(items):
            cur = (items[i] or "").strip()
            if (
                _is_heading_only_chunk(cur)
                and i + 1 < len(items)
                and len(items[i + 1]) <= int(chunk_size * 1.6)
            ):
                nxt = items[i + 1].strip()
                if not nxt.startswith(cur):
                    merged.append(f"{cur}\n{nxt}")
                else:
                    merged.append(nxt)
                i += 2
                continue
            merged.append(items[i])
            i += 1
        return merged

    return _merge_heading_only_chunks(chunks)


# ============================================================
# Estratégia 3: Semantic (corte por mudança de assunto)
# ============================================================

def split_semantic(text: str, embedder=None) -> List[str]:
    """Divide texto usando embeddings para detectar mudanças de assunto.
    
    Esta é a estratégia mais avançada. Em vez de cortar por tamanho
    ou estrutura, ela analisa o significado semântico de cada parágrafo.
    Quando detecta que o assunto mudou significativamente (o embedding
    de um parágrafo é muito diferente do próximo), corta ali.
    
    Exemplo:
      Parágrafo 1: "A mussarela é fabricada por filagem..."
      Parágrafo 2: "A temperatura ideal de filagem é 80°C..."
      Parágrafo 3: "A legislação IN 30 define os padrões..."
      
    O semantic chunker detecta que os parágrafos 1-2 são sobre o mesmo
    tema (fabricação) e o parágrafo 3 mudou de assunto (legislação),
    então corta entre 2 e 3.
    
    Trade-off:
      + Chunks mais coerentes (cada chunk é sobre um assunto)
      - Mais lento (precisa gerar embeddings de cada parágrafo)
      - Mais caro (chamadas extras à API de embeddings)
      - Tamanho dos chunks é imprevisível
    
    Requer:
      - Um dos pacotes que expõe SemanticChunker (ver importação acima)
      - Um embedder válido (ex: OpenAIEmbeddings)
    
    Código idêntico ao original (loaders.py linhas 65-76).
    """
    if not _HAS_SEMANTIC:
        raise RuntimeError(
            "SemanticChunker não está disponível. "
            "Instale langchain-experimental>=0.3.4 ou "
            "langchain-text-splitters>=0.3.9"
        )
    
    if embedder is None:
        raise RuntimeError(
            "Semantic chunking requer um 'embedder' válido. "
            "Passe OpenAIEmbeddings() como parâmetro."
        )
    
    # breakpoint_threshold_type="interquartile" define como o splitter
    # decide o limiar de "mudança de assunto":
    # - Calcula a distância entre embeddings de parágrafos consecutivos
    # - Usa o intervalo interquartil (IQR) para definir o que é "grande"
    # - Se a distância é maior que Q3 + 1.5*IQR, é um ponto de corte
    splitter = SemanticChunker(
        embedder,
        breakpoint_threshold_type="interquartile",
    )
    
    return splitter.split_text(text)


# ============================================================
# Função principal: escolhe a estratégia e divide
# ============================================================

def split_text(
    text: str,
    strategy: str = "fixed",
    embedder=None,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> Tuple[List[str], str]:
    """Divide texto em chunks conforme a estratégia escolhida.
    
    Esta é a função que os outros módulos chamam. Ela recebe o texto
    e a estratégia, e delega para a função correta.
    
    Parâmetros:
        text: Texto completo a ser dividido.
        strategy: "fixed", "markdown" ou "semantic".
        embedder: Necessário apenas para strategy="semantic".
        chunk_size: Tamanho do chunk (ignorado para "semantic").
        chunk_overlap: Sobreposição (ignorado para "semantic").
    
    Retorna:
        Tupla (chunks, strategy_usada).
        A strategy_usada pode ser diferente da pedida se houver
        fallback (ex: "semantic" sem embedder → erro, não fallback).
    
    Diferença do original:
    O original (loaders.py linhas 79-94) é idêntico na lógica.
    Aqui adicionamos validação do parâmetro strategy com mensagem
    de erro mais clara.
    """
    # Normaliza o nome da estratégia (case-insensitive)
    s = (strategy or "").strip().lower()
    
    if not s:
        raise RuntimeError(
            "Estratégia de chunking não informada. "
            "Use 'fixed', 'markdown' ou 'semantic'."
        )
    
    if s == "fixed":
        return split_fixed(text, chunk_size, chunk_overlap), "fixed"
    
    if s == "markdown":
        return split_markdown(text, chunk_size, chunk_overlap), "markdown"
    
    if s == "semantic":
        return split_semantic(text, embedder), "semantic"
    
    # Sem fallback: se a estratégia é inválida, falha explicitamente.
    # O original também faz isso (loaders.py linha 93).
    # Melhor falhar cedo do que processar com a estratégia errada.
    raise RuntimeError(
        f"Estratégia de chunking inválida: '{strategy}'. "
        f"Use 'fixed', 'markdown' ou 'semantic'."
    )


# ============================================================
# Função utilitária: divide com config por tipo de documento
# ============================================================

def split_by_doc_type(
    text: str,
    doc_type: str,
    strategy: str = "markdown",
    embedder=None,
) -> Tuple[List[str], str]:
    """Divide texto usando tamanhos de chunk do config.py baseados no doc_type.
    
    Esta função NÃO existe no projeto original — é uma adição para
    o cenário de laticínios, onde diferentes tipos de documento
    precisam de tamanhos diferentes.
    
    Em vez de passar chunk_size manualmente, você informa o doc_type
    ("legislacao", "manual", "faq", etc.) e a função busca o tamanho
    ideal no dicionário CHUNK_SIZES do config.py.
    
    Exemplo:
        split_by_doc_type(text, "legislacao")
        → usa chunk_size=600, chunk_overlap=100
        
        split_by_doc_type(text, "manual")
        → usa chunk_size=1200, chunk_overlap=250
        
        split_by_doc_type(text, "tipo_desconhecido")
        → usa DEFAULT_CHUNK_SIZE=1000, DEFAULT_CHUNK_OVERLAP=200
    
    Parâmetros:
        text: Texto completo.
        doc_type: Tipo do documento (deve corresponder a uma chave
                  em CHUNK_SIZES do config.py).
        strategy: Estratégia de chunking ("fixed", "markdown", "semantic").
        embedder: Necessário apenas para "semantic".
    
    Retorna:
        Tupla (chunks, strategy_usada).
    
    Usado em: ingest.py → ao processar documentos do pipeline de ingestão.
    """
    from app.config import CHUNK_SIZES, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
    
    # Busca o tamanho configurado para este tipo de documento
    # Se não encontrar, usa o padrão
    size, overlap = CHUNK_SIZES.get(
        doc_type,
        (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP),
    )
    
    strategy_norm = (strategy or "").strip().lower()
    doc_type_norm = (doc_type or "").strip().lower()

    # Regra específica para legislação:
    # - Prioriza manter artigo + parágrafos juntos
    # - Ainda impõe teto absoluto para evitar chunks excessivos
    if strategy_norm == "markdown":
        max_ratio = 1.5
        max_chars: Optional[int] = None
        if doc_type_norm == "legislacao":
            max_ratio = 2.4
            max_chars = 1600
        return (
            split_markdown(
                text,
                chunk_size=size,
                chunk_overlap=overlap,
                max_section_ratio=max_ratio,
                max_section_chars=max_chars,
            ),
            "markdown",
        )

    return split_text(
        text,
        strategy=strategy,
        embedder=embedder,
        chunk_size=size,
        chunk_overlap=overlap,
    )
