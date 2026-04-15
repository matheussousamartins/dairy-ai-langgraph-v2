"""
tests/conftest.py — Fixtures compartilhadas para todos os testes

Adaptado do conftest.py do projeto original (tests/conftest.py).
Mudanças em relação ao original:
  - Usa SUPABASE_DB_URL e HETZNER_DB_URL em vez de DB_USER/DB_HOST/etc.
  - Remove fixtures de CRM (clean_db, unique_email)
  - Adiciona fixtures específicas para RAG de laticínios (db_supabase, db_hetzner)
  - Mantém llm_judge_yesno idêntico ao original (funciona perfeitamente)
  - Adiciona fixture agent_id para parametrizar testes por agente

Fixtures disponíveis:
  db_supabase       → connection string do Supabase (vector store)
  db_hetzner        → connection string do Hetzner (memory + logs)
  require_openai    → pula teste se OPENAI_API_KEY não está configurada
  require_cohere    → pula teste se COHERE_API_KEY não está configurada
  llm_judge_yesno   → juiz LLM que avalia se contextos respondem a pergunta
  rag_dataset        → carrega o dataset de perguntas (tests/fixtures/rag/rag_queries.yaml)
"""

import os
import pytest
import psycopg
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv


# ============================================================
# Carregamento do .env
# ============================================================

def _load_env() -> None:
    """Carrega variáveis do .env do projeto."""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)
    load_dotenv(override=False)


@pytest.fixture(scope="session", autouse=True)
def init_db_pools_for_tests():
    """Inicializa pools de conexão para toda a sessão de testes."""
    _load_env()
    from app.db.connection import init_pools, close_pools
    init_pools()
    yield
    close_pools()


# ============================================================
# Fixtures de banco de dados
# ============================================================

@pytest.fixture(scope="session")
def db_supabase():
    """Retorna a connection string do Supabase.
    
    Verifica se o banco está acessível. Se não, pula os testes.
    Aplica os scripts SQL (schema, índices) uma vez por sessão.
    
    Equivalente ao db_ready do original, mas para o Supabase.
    """
    _load_env()
    url = os.getenv("SUPABASE_DB_URL")
    if not url:
        pytest.skip("SUPABASE_DB_URL não configurada no .env")
    
    # Testa conexão
    try:
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as e:
        pytest.skip(f"Supabase inacessível: {e}")
    
    # Aplica schema e índices (idempotente)
    sql_dir = Path(__file__).resolve().parents[1] / "sql"
    sql_files = [
        sql_dir / "01_kb_schema.sql",
        sql_dir / "02_kb_indexes.sql",
    ]
    
    for sql_file in sql_files:
        if sql_file.exists():
            try:
                sql = sql_file.read_text(encoding="utf-8")
                with psycopg.connect(url) as conn:
                    conn.autocommit = True
                    with conn.cursor() as cur:
                        cur.execute(sql)
            except Exception as e:
                print(f"[conftest] Aviso ao aplicar {sql_file.name}: {e}")
    
    return url


@pytest.fixture(scope="session")
def db_hetzner():
    """Retorna a connection string do Hetzner.
    
    Verifica se o banco está acessível. Se não, pula os testes.
    Aplica as tabelas do app (chat_memories, logs) uma vez.
    """
    _load_env()
    url = os.getenv("HETZNER_DB_URL")
    if not url:
        pytest.skip("HETZNER_DB_URL não configurada no .env")
    
    try:
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as e:
        pytest.skip(f"Hetzner inacessível: {e}")
    
    # Aplica tabelas do app
    sql_file = Path(__file__).resolve().parents[1] / "sql" / "03_app_tables.sql"
    if sql_file.exists():
        try:
            sql = sql_file.read_text(encoding="utf-8")
            with psycopg.connect(url) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql)
        except Exception as e:
            print(f"[conftest] Aviso ao aplicar app_tables: {e}")
    
    return url


# ============================================================
# Fixtures de API keys
# ============================================================

@pytest.fixture(scope="session")
def require_openai():
    """Garante que OPENAI_API_KEY está configurada.
    
    Idêntica ao original (conftest.py linhas 94-100).
    Testes que chamam a API da OpenAI (embeddings, LLM) devem
    usar esta fixture para pular automaticamente se a key não existir.
    """
    _load_env()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY não configurada")
    return True


@pytest.fixture(scope="session")
def require_cohere():
    """Garante que COHERE_API_KEY está configurada.
    
    Idêntica ao original (conftest.py linhas 103-109).
    """
    _load_env()
    if not os.getenv("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY não configurada")
    return True


# ============================================================
# Juiz LLM (avaliador automático de qualidade)
# ============================================================

@pytest.fixture(scope="session")
def llm_judge_yesno():
    """Retorna um callable que usa LLM para julgar relevância.
    
    IDÊNTICO ao original (conftest.py linhas 112-134).
    
    O juiz recebe:
      - question: a pergunta do usuário
      - expected: a resposta esperada (referência)
      - contexts: lista de chunks ou respostas a avaliar
    
    Retorna True se os contextos são suficientes para responder
    a questão de acordo com o esperado. False caso contrário.
    
    Usa um LLM (gpt-4.1 ou o que estiver em EVAL_LLM_MODEL)
    como "juiz imparcial" — ele não sabe qual estratégia gerou
    os contextos, apenas avalia se são relevantes.
    
    Custo: ~$0.01 por avaliação.
    """
    _load_env()
    from langchain_openai import ChatOpenAI
    
    def _judge(question: str, expected: str, contexts: List[str]) -> bool:
        joined = "\n\n---\n\n".join(contexts[:3])
        prompt = (
            "Você é um avaliador especialista em tecnologia de laticínios. "
            "Analise os CONTEXTOS abaixo e responda apenas com 'sim' ou 'não' "
            "se eles são suficientes para responder à QUESTÃO e se contêm "
            "(direta ou claramente) o conceito/termo ESPERADO.\n\n"
            f"QUESTÃO: {question}\n"
            f"ESPERADO: {expected}\n\n"
            f"CONTEXTOS:\n{joined}\n\n"
            "Saída: apenas 'sim' ou 'não'."
        )
        llm = ChatOpenAI(
            model=os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
        out = (llm.invoke(prompt).content or "").strip().lower()
        return out.startswith("sim")
    
    return _judge


# ============================================================
# Dataset de perguntas
# ============================================================

@pytest.fixture(scope="session")
def rag_dataset() -> List[Dict[str, Any]]:
    """Carrega o dataset de perguntas do rag_queries.yaml.
    
    Retorna a lista de queries com agent_id, pergunta, expected, etc.
    Cada teste pode filtrar por agent_id ou group conforme necessário.
    """
    import yaml
    
    ds_file = Path(__file__).resolve().parent / "fixtures" / "rag" / "rag_queries.yaml"
    if not ds_file.exists():
        pytest.skip("Arquivo rag_queries.yaml não encontrado")
    
    data = yaml.safe_load(ds_file.read_text(encoding="utf-8")) or {}
    queries = data.get("queries") or []
    
    if not queries:
        pytest.skip("Dataset vazio em rag_queries.yaml")
    
    return queries


# ============================================================
# Fixture de amostragem (fast mode)
# ============================================================

@pytest.fixture(scope="session")
def max_queries() -> int | None:
    """Retorna o número máximo de queries para modo rápido.
    
    Controlado por variáveis de ambiente:
      RAG_FAST=1 → limita a 10 perguntas
      RAG_MAX_Q=N → limita a N perguntas
      Nenhuma → sem limite (roda todas)
    
    Idêntico ao padrão do original (usado em vários testes).
    """
    if os.getenv("RAG_FAST", "").strip().lower() in ("1", "true", "yes"):
        return 10
    max_q = os.getenv("RAG_MAX_Q", "").strip()
    if max_q.isdigit():
        return int(max_q)
    return None
