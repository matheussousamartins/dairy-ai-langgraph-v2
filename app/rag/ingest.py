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

from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END

from app.config import (
    EMBEDDING_MODEL,
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    CHUNK_SIZES,
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
) -> int:
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
        Quantidade de chunks inseridos/atualizados.
    """
    if not chunks:
        return 0
    
    count = 0
    
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
                    """,
                    (chunk["content"], vec_lit, meta_json, content_hash),
                )
                count += 1
    
    return count


# ============================================================
# Pipeline principal de ingestão
# ============================================================

def _compute_document_hash(text: str) -> str:
    """Gera hash estável do documento completo para deduplicação."""
    normalized = text.replace("\r\n", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _find_existing_ingestion(
    *,
    table_name: str,
    agent_id: int,
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
                    SELECT id, source_filename, chunk_count, ingested_at, status
                    FROM ingested_documents
                    WHERE table_name = %s
                      AND agent_id = %s
                      AND file_hash = %s
                      AND status = 'ingested'
                    ORDER BY ingested_at DESC
                    LIMIT 1
                    """,
                    (table_name, agent_id, file_hash),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "source_filename": row[1],
                    "chunk_count": row[2],
                    "ingested_at": row[3],
                    "status": row[4],
                }
    except Exception:
        return None

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
    _strategy = strategy or DEFAULT_CHUNK_STRATEGY
    source_size_bytes = len(text.encode("utf-8"))
    file_hash = _compute_document_hash(text)

    # ---- Etapa 0: Deduplicação por documento ----
    existing = _find_existing_ingestion(
        table_name=table_name,
        agent_id=agent_id,
        file_hash=file_hash,
    )
    if existing:
        _log_ingestion(
            table_name=table_name,
            agent_id=agent_id,
            source=source,
            doc_type=doc_type,
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
            "table_name": table_name,
            "source": source,
            "doc_type": doc_type,
            "strategy": _strategy,
            "processing_time_ms": int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            ),
            "duplicate_of": existing,
            "file_hash": file_hash,
        }
    
    # ---- Etapa 1: Chunking ----
    # split_by_doc_type busca o tamanho ideal no CHUNK_SIZES do config
    # Ex: doc_type="legislacao" → chunk_size=600, overlap=100
    chunks_text, resolved_strategy = split_by_doc_type(
        text=text,
        doc_type=doc_type,
        strategy=_strategy,
    )
    
    if not chunks_text:
        return {
            "success": False,
            "error": "Nenhum chunk gerado. Texto vazio ou muito curto.",
            "chunks_created": 0,
        }
    
    # ---- Etapa 2: Embeddings ----
    # Gera embeddings para todos os chunks em uma chamada batch
    vectors = embed_texts(chunks_text)
    
    # Monta a lista de chunks com metadados para o upsert
    chunks_with_embeddings = []
    for i, (chunk_text, vector) in enumerate(zip(chunks_text, vectors)):
        chunks_with_embeddings.append({
            "content": chunk_text,
            "embedding": vector,
            "metadata": {
                "agent_id": agent_id,
                "source": source,
                "doc_type": doc_type,
                "chunk_index": i,
                "strategy": resolved_strategy,
                "ingested_at": datetime.utcnow().isoformat(),
            },
        })
    
    # ---- Etapa 3: Upsert no Supabase ----
    chunks_saved = upsert_chunks(table_name, chunks_with_embeddings)
    
    # ---- Etapa 4: Log de rastreabilidade (Hetzner) ----
    # Registra na tabela ingested_documents para saber:
    # - Quais documentos foram ingeridos
    # - Quando e em qual tabela
    # - Quantos chunks geraram
    _log_ingestion(
        table_name=table_name,
        agent_id=agent_id,
        source=source,
        doc_type=doc_type,
        chunk_count=chunks_saved,
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
        "chunks_created": chunks_saved,
        "table_name": table_name,
        "source": source,
        "doc_type": doc_type,
        "strategy": resolved_strategy,
        "processing_time_ms": elapsed_ms,
        "file_hash": file_hash,
    }


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
