"""
rag/ingest.py — Pipeline de ingestão de documentos

Este módulo processa documentos e armazena no Supabase como embeddings.
É o equivalente em código do pipeline de ingestão que construímos no N8N
(Form → Extract PDF → LLM Markdown → Chunking → Supabase).

No projeto original do curso (app/rag/ingest.py), o pipeline tem 2 etapas:
  1. stage_docs (staging): lê arquivos .md e salva em kb_docs
  2. chunk_docs (materialização): lê kb_docs, chunka, embeda, salva em kb_chunks

Aqui adaptamos para o cenário de laticínios com 3 etapas:
  1. Receber texto (já extraído do PDF e convertido em Markdown)
  2. Chunking adaptativo (tamanho varia por doc_type)
  3. Embedding + upsert na tabela do agente no Supabase

A etapa de "Extract PDF → LLM Markdown" acontece ANTES deste módulo:
  - No N8N: nós Extract from File + Gemini Markdown fazem isso
  - Em código: o webapp.py recebe o texto já processado
  - Futuramente: podemos adicionar extração de PDF aqui (PyMuPDF + LLM)

Diferenças em relação ao original:
  - Original usa 1 tabela (kb_chunks) com filtro por empresa
  - Adaptado usa 6 tabelas (1 por agente) sem filtro
  - Original tem staging (kb_docs) + materialização (kb_chunks)
  - Adaptado simplifica: recebe texto → chunka → embeda → salva direto
  - Original lê arquivos .md de um diretório
  - Adaptado recebe texto via API (webhook) ou de diretório
  - Adicionado: rastreabilidade na tabela ingested_documents (Hetzner)

Grafo LangGraph (opcional):
  O original expõe a ingestão como um grafo LangGraph (para usar no Studio).
  Mantemos essa capacidade aqui também, mas adicionamos a API via webhook
  como forma principal de ingestão.
"""

from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
from pathlib import Path
import hashlib
import re
import unicodedata
from uuid import uuid4

from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END

from app.config import (
    EMBEDDING_MODEL,
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    CHUNK_SIZES,
    INGEST_BLOCK_LOW_QUALITY,
    INGEST_MIN_TEXT_CHARS,
    INGEST_MIN_WORDS,
    INGEST_MIN_TEXT_CHARS_GLOSSARIO,
    INGEST_MIN_WORDS_GLOSSARIO,
    INGEST_MAX_GARBLED_RATIO,
    INGEST_MIN_QUALITY_SCORE,
    INGEST_MIN_CHUNK_ALNUM,
)
from app.rag.loaders import split_text, split_by_doc_type
from app.db.connection import get_supabase_conn, get_hetzner_conn


# ============================================================
# Geração de embeddings em lote
# ============================================================

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Gera embeddings para uma lista de textos.
    
    Processa todos os textos em uma única chamada à API da OpenAI
    (batch embedding). Isso é MUITO mais eficiente do que chamar
    a API uma vez por texto.
    
    Exemplo:
      - 50 chunks × 1 chamada por chunk = 50 chamadas (~10 segundos)
      - 50 chunks × 1 chamada batch = 1 chamada (~1 segundo)
    
    A API aceita até 2048 textos por chamada (ou 8MB de texto total).
    Para documentos grandes, o embed_documents do LangChain faz
    batching automático.
    
    Custo: ~$0.00002 por chunk. 1000 chunks ≈ $0.02.
    
    Função idêntica ao original (ingest.py linhas 26-29).
    """
    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return emb.embed_documents(texts)


# ============================================================
# Conversão de vetor para literal SQL
# ============================================================

def vec_to_literal(v: List[float]) -> str:
    """Converte lista de floats para formato pgvector.
    
    Idêntica à do search.py e do original (ingest.py linha 22).
    Duplicada aqui para manter o módulo independente
    (não importar do search.py evita dependência circular).
    """
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# ============================================================
# Upsert de chunks no Supabase
# ============================================================

def upsert_chunks(
    table_name: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Insere ou atualiza chunks com embeddings na tabela do Supabase.
    
    Cada chunk é um dict com:
      - content: texto do chunk
      - embedding: vetor de 1536 dimensões
      - metadata: dict com metadados (source, doc_type, agent_id, etc.)
    
    A operação é idempotente: se um chunk com o mesmo conteúdo já
    existir (baseado em hash do content), ele é atualizado em vez
    de duplicado. Isso permite re-ingerir documentos sem criar
    duplicatas na base.
    
    No projeto original (ingest.py linhas 32-62), o upsert usa
    (doc_path, chunk_ix) como chave de unicidade. Aqui usamos um
    hash SHA-256 do conteúdo, que é mais flexível — funciona mesmo
    quando o documento não vem de um arquivo (ex: texto colado no form).
    
    Parâmetros:
        table_name: Nome da tabela no Supabase
                    (ex: "embeddings_agente_1_queijos").
        chunks: Lista de chunks com content, embedding e metadata.
    
    Retorna:
        Dict com quantidade de chunks processados, inseridos e atualizados.
    """
    if not chunks:
        return {"processed": 0, "inserted": 0, "updated": 0}

    processed = 0
    inserted = 0
    updated = 0
    
    with get_supabase_conn() as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                # Gera um hash do conteúdo para usar como ID
                # Isso garante idempotência: mesmo texto = mesmo hash = UPDATE
                content_hash = hashlib.sha256(
                    chunk["content"].encode("utf-8")
                ).hexdigest()[:16]  # 16 chars do hash é suficiente
                
                vec_lit = vec_to_literal(chunk["embedding"])
                
                # metadata é armazenado como JSONB no Postgres
                # Permite buscas por metadados futuramente
                # Ex: SELECT * WHERE metadata->>'doc_type' = 'legislacao'
                import json
                meta_json = json.dumps(chunk.get("metadata", {}))
                
                # INSERT com ON CONFLICT para idempotência
                # Se já existe um chunk com esse content_hash na tabela,
                # atualiza o embedding e metadata (pode ter mudado se
                # re-processou o documento com outro modelo)
                #
                # A tabela precisa ter uma coluna content_hash com
                # UNIQUE constraint. Ver sql/01_kb_schema.sql.
                cur.execute(
                    f"""
                    INSERT INTO {table_name} 
                        (content, embedding, metadata, content_hash)
                    VALUES (%s, %s::vector, %s::jsonb, %s)
                    ON CONFLICT (content_hash) 
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        content = EXCLUDED.content
                    RETURNING (xmax = 0) AS was_inserted
                    """,
                    (chunk["content"], vec_lit, meta_json, content_hash),
                )
                row = cur.fetchone()
                was_inserted = bool(row[0]) if row else False
                processed += 1
                if was_inserted:
                    inserted += 1
                else:
                    updated += 1

    return {
        "processed": processed,
        "inserted": inserted,
        "updated": updated,
    }


# ============================================================
# Pipeline principal de ingestão
# ============================================================

def _compute_document_hash(text: str) -> str:
    """Gera hash estavel do documento completo para deduplicacao."""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_glossary_text(text: str) -> str:
    """Padroniza glossario tabular para formato canonico.

    Entrada esperada (comum): tabela markdown com colunas
    "Palavra Encontrada" e "Substituicao".
    Saida: linhas estaveis no formato
      termo: <x> | substituicao: <y>
    Isso melhora deduplicacao, chunking e qualidade semantica do embedding.
    """
    raw = unicodedata.normalize("NFKC", text or "")
    lines = [ln.strip() for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned = [ln for ln in lines if ln]
    if not cleaned:
        return ""

    normalized_lines: List[str] = []
    table_rows = [ln for ln in cleaned if "|" in ln]
    if table_rows:
        for ln in table_rows:
            # Ignora cabecalho/separador de tabela markdown
            low = ln.lower()
            if "palavra encontrada" in low and "substit" in low:
                continue
            if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?", ln):
                continue
            parts = [p.strip() for p in ln.strip("|").split("|")]
            if len(parts) < 2:
                continue
            termo = re.sub(r"\s+", " ", parts[0]).strip(" -;:.")
            sub = re.sub(r"\s+", " ", parts[1]).strip(" -;:.")
            if termo and sub:
                normalized_lines.append(f"termo: {termo} | substituicao: {sub}")
    else:
        # Suporta formato "Registro + bullets", ex:
        # ## Registro 001
        # - Palavra encontrada: Starter
        # - Substituição: Fermento
        current_term: Optional[str] = None
        current_sub: Optional[str] = None

        def _flush_pair() -> None:
            nonlocal current_term, current_sub
            if current_term and current_sub:
                normalized_lines.append(
                    f"termo: {current_term} | substituicao: {current_sub}"
                )
            current_term = None
            current_sub = None

        for ln in cleaned:
            ln2 = re.sub(r"\s+", " ", ln).strip()
            low = ln2.lower()

            # ignora cabeçalhos genéricos e de registro
            if low.startswith("#"):
                if "registro" in low:
                    _flush_pair()
                continue

            # remove marcador de lista no início
            ln2 = re.sub(r"^\s*[-*]\s*", "", ln2).strip()
            low2 = ln2.lower()

            # termo
            m_term = re.match(
                r"^(palavra encontrada|termo)\s*:\s*(.+)$",
                low2,
                flags=re.IGNORECASE,
            )
            if m_term:
                _flush_pair()
                current_term = re.sub(
                    r"^(palavra encontrada|termo)\s*:\s*",
                    "",
                    ln2,
                    flags=re.IGNORECASE,
                ).strip(" -;:.")
                continue

            # substituição
            m_sub = re.match(
                r"^(substituicao|substituição|equivalente)\s*:\s*(.+)$",
                low2,
                flags=re.IGNORECASE,
            )
            if m_sub:
                current_sub = re.sub(
                    r"^(substituicao|substituição|equivalente)\s*:\s*",
                    "",
                    ln2,
                    flags=re.IGNORECASE,
                ).strip(" -;:.")
                continue

            # fallback para linha tipo "A -> B" ou "A: B"
            if "->" in ln2:
                _flush_pair()
                parts = [p.strip(" -;:.") for p in ln2.split("->", 1)]
                if len(parts) == 2 and parts[0] and parts[1]:
                    normalized_lines.append(
                        f"termo: {parts[0]} | substituicao: {parts[1]}"
                    )
                continue

            if ":" in ln2 and not low2.startswith(("fonte:", "nota:")):
                maybe_term, maybe_sub = [p.strip(" -;:.") for p in ln2.split(":", 1)]
                if maybe_term and maybe_sub:
                    _flush_pair()
                    normalized_lines.append(
                        f"termo: {maybe_term} | substituicao: {maybe_sub}"
                    )

        _flush_pair()

        # fallback final: preserva linhas se nada foi extraído em pares
        if not normalized_lines:
            for ln in cleaned:
                ln2 = re.sub(r"\s+", " ", ln).strip()
                if ln2:
                    normalized_lines.append(ln2)

    deduped: List[str] = []
    seen = set()
    for ln in normalized_lines:
        k = ln.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(ln)

    if not deduped:
        return ""
    return "\n".join(deduped).strip() + "\n"


def _preprocess_text_by_doc_type(text: str, doc_type: str) -> str:
    """Pre-processa texto por tipo de documento para melhorar consistencia."""
    dt = (doc_type or "").strip().lower()
    if dt == "glossario":
        return _normalize_glossary_text(text)
    return text or ""


def _normalize_source_name(source: str) -> str:
    """Normaliza nome/origem para filtros de metadados consistentes."""
    raw = unicodedata.normalize("NFKC", (source or "").strip())
    raw = raw.replace("\\", "/")
    raw = raw.split("/")[-1]
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw.lower()


def _extract_legislation_chunk_metadata(chunk_text: str) -> Dict[str, Any]:
    """Extrai metadados estruturados de chunks legislativos."""
    meta: Dict[str, Any] = {}
    if not chunk_text:
        return meta

    first_line = chunk_text.split("\n", 1)[0].strip()
    path_parts = [p.strip() for p in re.split(r"\s*>\s*", first_line) if p.strip()]
    if path_parts:
        meta["section_path"] = " > ".join(path_parts)
        meta["section_leaf"] = path_parts[-1]

    art = re.search(r"\bArt\.\s*(\d+[A-Za-z]?)", chunk_text, flags=re.IGNORECASE)
    if art:
        meta["article_number"] = art.group(1)

    par = re.search(r"§\s*(\d+)", chunk_text)
    if par:
        meta["paragraph_number"] = par.group(1)

    if path_parts:
        leaf = path_parts[-1]
        m_title = re.match(
            r"Art\.\s*\d+[A-Za-zº°]*\s*[^A-Za-z0-9\s]?\s*(.+)$",
            leaf,
            flags=re.IGNORECASE,
        )
        if m_title:
            meta["article_title"] = re.sub(r"\s+", " ", m_title.group(1)).strip()
        elif leaf.lower().startswith("art."):
            for sep in ("—", "-", ":"):
                if sep in leaf:
                    maybe_title = leaf.split(sep, 1)[1].strip()
                    if maybe_title:
                        meta["article_title"] = re.sub(r"\s+", " ", maybe_title).strip()
                    break

    return meta


def _build_chunk_metadata(
    *,
    agent_id: int,
    source: str,
    doc_type: str,
    chunk_index: int,
    strategy: str,
    chunk_text: str,
) -> Dict[str, Any]:
    """Monta metadados do chunk para auditoria e filtros de recuperação."""
    normalized_chunk = chunk_text or ""
    words = re.findall(r"\b[\wÀ-ÿ]{2,}\b", normalized_chunk)
    metadata: Dict[str, Any] = {
        "agent_id": agent_id,
        "source": source,
        "source_norm": _normalize_source_name(source),
        "doc_type": doc_type,
        "chunk_index": chunk_index,
        "strategy": strategy,
        "chunk_chars": len(normalized_chunk),
        "chunk_words": len(words),
        "chunk_hash": hashlib.sha256(normalized_chunk.encode("utf-8")).hexdigest()[:16],
        "ingested_at": datetime.utcnow().isoformat(),
    }
    if (doc_type or "").strip().lower() == "legislacao":
        metadata.update(_extract_legislation_chunk_metadata(normalized_chunk))
    return metadata


def _assess_text_quality(text: str, doc_type: str = "manual") -> Dict[str, Any]:
    """Avalia qualidade minima do texto para ingestao."""
    raw = text or ""
    normalized = unicodedata.normalize("NFKC", raw)
    words = re.findall(r"\b[\wÀ-ÿ]{2,}\b", normalized)
    word_count = len(words)
    char_count = len(normalized)

    garbled_tokens = ["�", "Ã", "Â", "Ð", "Ñ", "\x00"]
    garbled_hits = sum(normalized.count(tok) for tok in garbled_tokens)
    garbled_ratio = garbled_hits / max(1, char_count)
    mojibake_tokens = [
        "Ã",
        "Â",
        "â€",
        "â€“",
        "â€”",
        "ï¿½",
    ]
    mojibake_hits = sum(normalized.count(tok) for tok in mojibake_tokens)
    mojibake_ratio = mojibake_hits / max(1, word_count)
    max_mojibake_ratio = 0.03

    dt = (doc_type or "").strip().lower()
    min_chars = INGEST_MIN_TEXT_CHARS
    min_words = INGEST_MIN_WORDS
    if dt == "glossario":
        min_chars = INGEST_MIN_TEXT_CHARS_GLOSSARIO
        min_words = INGEST_MIN_WORDS_GLOSSARIO

    length_score = min(1.0, char_count / max(1, min_chars))
    words_score = min(1.0, word_count / max(1, min_words))
    garble_score = max(0.0, 1.0 - (garbled_ratio / max(1e-9, INGEST_MAX_GARBLED_RATIO)))
    quality_score = round((0.35 * length_score + 0.35 * words_score + 0.30 * garble_score) * 100, 2)

    reasons: List[str] = []
    if char_count < min_chars:
        reasons.append(f"Texto muito curto ({char_count} chars < {min_chars})")
    if word_count < min_words:
        reasons.append(f"Poucas palavras ({word_count} < {min_words})")
    if garbled_ratio > INGEST_MAX_GARBLED_RATIO:
        reasons.append(
            f"Alta taxa de caracteres suspeitos ({garbled_ratio:.2%} > {INGEST_MAX_GARBLED_RATIO:.2%})"
        )
    if mojibake_ratio > max_mojibake_ratio:
        reasons.append(
            f"Texto com sinais fortes de encoding corrompido/mojibake ({mojibake_ratio:.2%} > {max_mojibake_ratio:.2%})"
        )
    if quality_score < INGEST_MIN_QUALITY_SCORE:
        reasons.append(f"Quality score abaixo do minimo ({quality_score} < {INGEST_MIN_QUALITY_SCORE})")

    return {
        "quality_gate_passed": len(reasons) == 0,
        "quality_score": quality_score,
        "text_chars": char_count,
        "word_count": word_count,
        "garbled_ratio": round(garbled_ratio, 6),
        "mojibake_ratio": round(mojibake_ratio, 6),
        "thresholds": {
            "min_text_chars": min_chars,
            "min_words": min_words,
            "max_garbled_ratio": INGEST_MAX_GARBLED_RATIO,
            "max_mojibake_ratio": max_mojibake_ratio,
            "min_quality_score": INGEST_MIN_QUALITY_SCORE,
        },
        "quality_issues": reasons,
    }


def _find_existing_ingestion(
    *,
    file_hash: str,
) -> Optional[Dict[str, Any]]:
    """Busca documento já ingerido por hash na tabela de rastreabilidade.

    Compatível com esquema antigo: se a coluna file_hash não existir ainda,
    retorna None e segue ingestão normalmente.
    """
    try:
        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, table_name, agent_id, source_filename, chunk_count, ingested_at, status
                    FROM ingested_documents
                    WHERE file_hash = %s
                      AND status = 'ingested'
                    ORDER BY ingested_at DESC
                    LIMIT 1
                    """,
                    (file_hash,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "table_name": row[1],
                    "agent_id": row[2],
                    "source_filename": row[3],
                    "chunk_count": row[4],
                    "ingested_at": row[5],
                    "status": row[6],
                }
    except Exception:
        return None


def _reserve_ingestion_slot(
    *,
    table_name: str,
    agent_id: int,
    source: str,
    doc_type: str,
    file_hash: str,
    source_size_bytes: Optional[int],
) -> Dict[str, Any]:
    """Reserva um slot de ingestao para evitar corrida entre uploads simultaneos.

    Requer indice unico parcial em `ingested_documents(file_hash)` para status
    ativos (`processing`/`ingested`). Se o indice ainda nao existir, o fallback
    e seguir com o comportamento antigo.
    """
    token = f"ingest-{uuid4().hex}"
    try:
        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingested_documents
                        (table_name, agent_id, source_filename, doc_type, chunk_count,
                         status, ingested_at, file_hash, source_size_bytes, status_detail)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING id
                    """,
                    (
                        table_name,
                        agent_id,
                        source,
                        doc_type,
                        0,
                        "processing",
                        datetime.utcnow(),
                        file_hash,
                        source_size_bytes,
                        f"reservation_token={token}",
                    ),
                )
                row = cur.fetchone()
                if row:
                    return {"reserved": True, "reservation_id": row[0], "reservation_token": token}

                cur.execute(
                    """
                    SELECT id, table_name, agent_id, source_filename, chunk_count, ingested_at, status
                    FROM ingested_documents
                    WHERE file_hash = %s
                      AND status IN ('processing', 'ingested')
                    ORDER BY ingested_at DESC
                    LIMIT 1
                    """,
                    (file_hash,),
                )
                existing = cur.fetchone()
                if existing:
                    return {
                        "reserved": False,
                        "existing": {
                            "id": existing[0],
                            "table_name": existing[1],
                            "agent_id": existing[2],
                            "source_filename": existing[3],
                            "chunk_count": existing[4],
                            "ingested_at": existing[5],
                            "status": existing[6],
                        },
                    }
                return {"reserved": True, "reservation_id": None, "reservation_token": token}
    except Exception:
        return {"reserved": True, "reservation_id": None, "reservation_token": token}


def _update_ingestion_status(
    *,
    reservation_id: Optional[int],
    reservation_token: Optional[str],
    status: str,
    chunk_count: int,
    status_detail: Optional[str] = None,
) -> None:
    """Atualiza o registro reservado de ingestao para status final."""
    if not reservation_id and not reservation_token:
        return
    try:
        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                if reservation_id:
                    cur.execute(
                        """
                        UPDATE ingested_documents
                        SET status = %s,
                            chunk_count = %s,
                            status_detail = %s,
                            ingested_at = %s
                        WHERE id = %s
                        """,
                        (status, chunk_count, status_detail, datetime.utcnow(), reservation_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE ingested_documents
                        SET status = %s,
                            chunk_count = %s,
                            status_detail = %s,
                            ingested_at = %s
                        WHERE status_detail = %s
                        """,
                        (
                            status,
                            chunk_count,
                            status_detail,
                            datetime.utcnow(),
                            f"reservation_token={reservation_token}",
                        ),
                    )
    except Exception as e:
        print(f"[ingest] Aviso: falha ao atualizar status de ingestao: {e}")

def ingest_text(
    text: str,
    table_name: str,
    agent_id: int,
    source: str = "upload",
    doc_type: str = "manual",
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Pipeline completo: texto → chunks → embeddings → Supabase.
    
    Esta é a função principal que o webapp.py chama quando recebe
    um documento via webhook ou form.
    
    Etapas:
      1. Divide o texto em chunks (tamanho adaptativo por doc_type)
      2. Gera embeddings para todos os chunks (batch)
      3. Insere os chunks com embeddings no Supabase
      4. Registra a ingestão no Hetzner (rastreabilidade)
      5. Retorna estatísticas (chunks gerados, tempo, etc.)
    
    Parâmetros:
        text: Texto completo do documento (já em Markdown limpo).
              Não aceita PDF binário — a extração deve ser feita antes.
        table_name: Tabela destino no Supabase
                    (ex: "embeddings_agente_1_queijos").
        agent_id: ID do agente (1-6). Usado nos metadados e logs.
        source: Nome/origem do documento (ex: "IN-76-leite-cru.pdf").
                Salvo nos metadados para rastreabilidade.
        doc_type: Tipo do documento ("legislacao", "manual", "faq", etc.).
                  Determina o tamanho do chunk via config.py.
        strategy: Estratégia de chunking. None = usa DEFAULT_CHUNK_STRATEGY.
    
    Retorna:
        Dict com estatísticas:
        {
            "success": True,
            "chunks_created": 45,
            "table_name": "embeddings_agente_1_queijos",
            "source": "IN-76-leite-cru.pdf",
            "doc_type": "legislacao",
            "strategy": "markdown",
            "processing_time_ms": 3200,
        }
    
    Exemplo de uso:
        result = ingest_text(
            text="# Instrução Normativa 76\n\nArt. 1º...",
            table_name="embeddings_agente_3_regulatorios",
            agent_id=3,
            source="IN-76-2018.pdf",
            doc_type="legislacao",
        )
    """
    start_time = datetime.utcnow()
    normalized_doc_type = (doc_type or "").strip().lower()
    _strategy = strategy or DEFAULT_CHUNK_STRATEGY
    if normalized_doc_type == "glossario":
        _strategy = "fixed"

    text = _preprocess_text_by_doc_type(text, normalized_doc_type)
    source_size_bytes = len(text.encode("utf-8"))
    quality_report = _assess_text_quality(text, doc_type=normalized_doc_type)
    reservation_id: Optional[int] = None
    reservation_token: Optional[str] = None

    if INGEST_BLOCK_LOW_QUALITY and not quality_report["quality_gate_passed"]:
        _log_ingestion(
            table_name=table_name,
            agent_id=agent_id,
            source=source,
            doc_type=normalized_doc_type,
            chunk_count=0,
            status="rejected_quality",
            file_hash=None,
            source_size_bytes=source_size_bytes,
            status_detail="; ".join(quality_report.get("quality_issues", []))[:1000],
        )
        return {
            "success": False,
            "error": "Documento bloqueado pelo quality gate de ingestao.",
            "chunks_created": 0,
            "chunks_processed": 0,
            "chunks_inserted": 0,
            "chunks_updated": 0,
            "table_name": table_name,
            "agent_id": agent_id,
            "source": source,
            "doc_type": normalized_doc_type,
            "strategy": _strategy,
            "processing_time_ms": int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            ),
            **quality_report,
        }

    file_hash = _compute_document_hash(text)

    # ---- Etapa 0: Reserva de ingestao + deduplicacao por documento ----
    reservation = _reserve_ingestion_slot(
        table_name=table_name,
        agent_id=agent_id,
        source=source,
        doc_type=normalized_doc_type,
        file_hash=file_hash,
        source_size_bytes=source_size_bytes,
    )
    reservation_id = reservation.get("reservation_id")
    reservation_token = reservation.get("reservation_token")

    existing = reservation.get("existing")
    if existing:
        _log_ingestion(
            table_name=table_name,
            agent_id=agent_id,
            source=source,
            doc_type=normalized_doc_type,
            chunk_count=0,
            status="skipped_duplicate",
            file_hash=file_hash,
            source_size_bytes=source_size_bytes,
            status_detail=(
                f"Documento já ingerido como {existing['source_filename']} "
                f"(id={existing['id']}) em {existing['ingested_at']}"
            ),
        )
        return {
            "success": True,
            "skipped_duplicate": True,
            "chunks_created": 0,
            "chunks_processed": 0,
            "chunks_inserted": 0,
            "chunks_updated": 0,
            "table_name": table_name,
            "agent_id": agent_id,
            "source": source,
            "doc_type": normalized_doc_type,
            "strategy": _strategy,
            "processing_time_ms": int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            ),
            "duplicate_of": existing,
            "file_hash": file_hash,
            **quality_report,
        }

    try:
        # ---- Etapa 1: Chunking ----
        # split_by_doc_type busca o tamanho ideal no CHUNK_SIZES do config
        # Ex: doc_type="legislacao" -> chunk_size=600, overlap=100
        chunks_text, resolved_strategy = split_by_doc_type(
            text=text,
            doc_type=normalized_doc_type,
            strategy=_strategy,
        )

        if not chunks_text:
            _update_ingestion_status(
                reservation_id=reservation_id,
                reservation_token=reservation_token,
                status="failed",
                chunk_count=0,
                status_detail="Nenhum chunk gerado. Texto vazio ou muito curto.",
            )
            return {
                "success": False,
                "error": "Nenhum chunk gerado. Texto vazio ou muito curto.",
                "chunks_created": 0,
                "chunks_processed": 0,
                "chunks_inserted": 0,
                "chunks_updated": 0,
                **quality_report,
            }

        # Filtra chunks degenerados (ex: ".", linha em branco, artefato de PDF)
        chunks_text = [
            c for c in chunks_text
            if sum(1 for ch in c if ch.isalnum()) >= INGEST_MIN_CHUNK_ALNUM
        ]

        # ---- Etapa 2: Embeddings ----
        # Gera embeddings para todos os chunks em uma chamada batch
        vectors = embed_texts(chunks_text)

        # Monta a lista de chunks com metadados para o upsert
        chunks_with_embeddings = []
        for i, (chunk_text, vector) in enumerate(zip(chunks_text, vectors)):
            chunks_with_embeddings.append({
                "content": chunk_text,
                "embedding": vector,
                "metadata": _build_chunk_metadata(
                    agent_id=agent_id,
                    source=source,
                    doc_type=normalized_doc_type,
                    chunk_index=i,
                    strategy=resolved_strategy,
                    chunk_text=chunk_text,
                ),
            })

        # ---- Etapa 3: Upsert no Supabase ----
        upsert_stats = upsert_chunks(table_name, chunks_with_embeddings)

        # ---- Etapa 4: Finaliza rastreabilidade (Hetzner) ----
        _update_ingestion_status(
            reservation_id=reservation_id,
            reservation_token=reservation_token,
            status="ingested",
            chunk_count=upsert_stats["processed"],
            status_detail=(
                f"inserted={upsert_stats['inserted']};updated={upsert_stats['updated']}"
            ),
        )

        # Se nao conseguiu reservar (ambiente sem migracao), mantem log classico
        if not reservation_id and not reservation_token:
            _log_ingestion(
                table_name=table_name,
                agent_id=agent_id,
                source=source,
                doc_type=normalized_doc_type,
                chunk_count=upsert_stats["processed"],
                status="ingested",
                file_hash=file_hash,
                source_size_bytes=source_size_bytes,
            )

        # ---- Etapa 5: Retorna estatísticas ----
        elapsed_ms = int(
            (datetime.utcnow() - start_time).total_seconds() * 1000
        )

        return {
            "success": True,
            "chunks_created": upsert_stats["processed"],
            "chunks_processed": upsert_stats["processed"],
            "chunks_inserted": upsert_stats["inserted"],
            "chunks_updated": upsert_stats["updated"],
            "table_name": table_name,
            "agent_id": agent_id,
            "source": source,
            "doc_type": normalized_doc_type,
            "strategy": resolved_strategy,
            "processing_time_ms": elapsed_ms,
            "file_hash": file_hash,
            **quality_report,
        }
    except Exception as e:
        _update_ingestion_status(
            reservation_id=reservation_id,
            reservation_token=reservation_token,
            status="failed",
            chunk_count=0,
            status_detail=f"Erro na ingestao: {str(e)[:900]}",
        )
        raise


def _log_ingestion(
    table_name: str,
    agent_id: int,
    source: str,
    doc_type: str,
    chunk_count: int,
    status: str = "ingested",
    file_hash: Optional[str] = None,
    source_size_bytes: Optional[int] = None,
    status_detail: Optional[str] = None,
) -> None:
    """Registra a ingestão na tabela ingested_documents (Hetzner).
    
    Equivalente ao nó "Log Ingestão" do pipeline N8N.
    Permite rastrear quais documentos foram ingeridos, quando,
    e quantos chunks geraram.
    
    Se o log falhar (ex: Hetzner offline), a ingestão NÃO é
    revertida — os chunks já foram salvos no Supabase. O log
    é informativo, não transacional.
    """
    try:
        with get_hetzner_conn() as conn:
            with conn.cursor() as cur:
                try:
                    # Esquema novo (com colunas de dedup e auditoria)
                    cur.execute(
                        """
                        INSERT INTO ingested_documents 
                            (table_name, agent_id, source_filename, doc_type,
                             chunk_count, status, ingested_at, file_hash,
                             source_size_bytes, status_detail)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            table_name,
                            agent_id,
                            source,
                            doc_type,
                            chunk_count,
                            status,
                            datetime.utcnow(),
                            file_hash,
                            source_size_bytes,
                            status_detail,
                        ),
                    )
                except Exception:
                    # Fallback para esquema antigo (sem novas colunas)
                    cur.execute(
                        """
                        INSERT INTO ingested_documents 
                            (table_name, agent_id, source_filename, doc_type,
                             chunk_count, status, ingested_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            table_name,
                            agent_id,
                            source,
                            doc_type,
                            chunk_count,
                            status,
                            datetime.utcnow(),
                        ),
                    )
    except Exception as e:
        # Log falhou mas não interrompe o fluxo
        # Em produção, isso deveria ir para um logger (não print)
        print(f"[ingest] Aviso: falha ao registrar log de ingestão: {e}")


# ============================================================
# Ingestão de diretório (múltiplos arquivos .md)
# ============================================================

def ingest_directory(
    base_dir: str,
    table_name: str,
    agent_id: int,
    doc_type: str = "manual",
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingere todos os arquivos .md de um diretório.
    
    Útil para ingestão em lote: coloca todos os PDFs convertidos
    em Markdown em uma pasta e roda este comando.
    
    Parâmetros:
        base_dir: Caminho do diretório com arquivos .md
        table_name: Tabela destino no Supabase
        agent_id: ID do agente
        doc_type: Tipo padrão para todos os arquivos
        strategy: Estratégia de chunking
    
    Retorna:
        Dict com total de arquivos e chunks processados.
    
    Exemplo:
        result = ingest_directory(
            base_dir="./docs/agente-1-queijos/md",
            table_name="embeddings_agente_1_queijos",
            agent_id=1,
            doc_type="manual",
        )
    """
    base = Path(base_dir)
    files = sorted([p for p in base.rglob("*.md") if p.is_file()])
    
    if not files:
        return {
            "success": False,
            "error": f"Nenhum arquivo .md encontrado em {base_dir}",
            "files_processed": 0,
            "total_chunks": 0,
        }
    
    total_chunks = 0
    files_processed = 0
    errors = []
    
    for filepath in files:
        text = filepath.read_text(encoding="utf-8")
        
        if not text.strip():
            errors.append(f"{filepath.name}: arquivo vazio")
            continue
        
        result = ingest_text(
            text=text,
            table_name=table_name,
            agent_id=agent_id,
            source=filepath.name,
            doc_type=doc_type,
            strategy=strategy,
        )
        
        if result.get("success"):
            total_chunks += result["chunks_created"]
            files_processed += 1
        else:
            errors.append(f"{filepath.name}: {result.get('error', 'erro desconhecido')}")
    
    return {
        "success": files_processed > 0,
        "files_processed": files_processed,
        "total_chunks": total_chunks,
        "errors": errors if errors else None,
    }


# ============================================================
# Grafo LangGraph para ingestão (compatibilidade com Studio)
# ============================================================
# 
# O projeto original expõe a ingestão como grafo LangGraph para
# poder ser executada no LangGraph Studio (interface visual).
# Mantemos essa capacidade para debugging e testes.

class IngestState(TypedDict, total=False):
    """Estado do grafo de ingestão.
    
    Campos que o Studio ou chamador externo pode definir:
      - text: texto do documento (obrigatório)
      - table_name: tabela destino (obrigatório)
      - agent_id: ID do agente
      - source: nome do documento
      - doc_type: tipo (legislacao, manual, faq, etc.)
      - strategy: estratégia de chunking
    
    Campos preenchidos pela execução:
      - chunks_created: quantos chunks foram gerados
      - success: se a ingestão foi bem sucedida
      - error: mensagem de erro (se falhou)
    """
    text: str
    table_name: str
    agent_id: int
    source: str
    doc_type: str
    strategy: str
    # Outputs
    chunks_created: int
    success: bool
    error: str


def _node_ingest(state: IngestState) -> IngestState:
    """Nó único do grafo: executa o pipeline completo.
    
    Lê os parâmetros do estado, chama ingest_text(), e retorna
    os resultados no estado.
    """
    text = state.get("text", "")
    table_name = state.get("table_name", "")
    
    if not text or not table_name:
        return {
            "success": False,
            "error": "Campos 'text' e 'table_name' são obrigatórios.",
            "chunks_created": 0,
        }
    
    result = ingest_text(
        text=text,
        table_name=table_name,
        agent_id=state.get("agent_id", 0),
        source=state.get("source", "studio"),
        doc_type=state.get("doc_type", "manual"),
        strategy=state.get("strategy"),
    )
    
    return {
        "success": result.get("success", False),
        "chunks_created": result.get("chunks_created", 0),
        "error": result.get("error", ""),
    }


def compile_ingest_graph():
    """Compila o grafo de ingestão para uso no LangGraph Studio.
    
    O grafo tem apenas 1 nó (ingest) porque toda a lógica está
    encapsulada na função ingest_text(). No original, são 2 nós
    (stage + chunk) porque o staging e a materialização são separados.
    Aqui simplificamos porque não temos staging intermediário.
    """
    g = StateGraph(IngestState)
    g.add_node("ingest", _node_ingest)
    g.set_entry_point("ingest")
    g.set_finish_point("ingest")
    return g.compile()


# Grafo compilado — importável pelo langgraph.json
graph = compile_ingest_graph()
