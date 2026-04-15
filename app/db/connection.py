"""
db/connection.py — Pool de conexões Postgres

Este arquivo gerencia as conexões com os dois bancos de dados do projeto:
  - Supabase (vector store): tabelas de embeddings, funções de busca
  - Hetzner (operacional): chat_memories, interaction_logs, users

No projeto original do curso, existe apenas uma função get_conn() em
app/agent/tools.py que cria uma conexão nova a cada chamada:

    def get_conn():
        return psycopg.connect(
            f"host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')} ..."
        )

Isso funciona para testes, mas em produção é ineficiente:
  - Cada request abre uma conexão TCP nova (lento: ~50-100ms por conexão)
  - Sob carga (10 usuários simultâneos), o banco recusa conexões

A solução é usar um CONNECTION POOL: um conjunto de conexões pré-abertas
que são reutilizadas entre requests. O pool abre N conexões no startup e
quando um módulo precisa de uma conexão, pega uma emprestada do pool.
Quando termina, devolve ao pool (não fecha). Isso elimina o overhead de
abrir/fechar conexões a cada request.

Usamos o psycopg_pool.ConnectionPool, que é o pool oficial do psycopg 3.
Ele gerencia automaticamente:
  - Abertura de conexões no startup
  - Reutilização entre requests
  - Reconexão automática se uma conexão cair
  - Limite máximo de conexões simultâneas
"""

import psycopg
from psycopg_pool import ConnectionPool
from contextlib import contextmanager
from typing import Generator

from app.config import SUPABASE_DB_URL, HETZNER_DB_URL


# ============================================================
# Pools de conexão (inicializados no startup do servidor)
# ============================================================

# Pool para o SUPABASE (vector store).
# min_size=2: mantém 2 conexões abertas sempre (para buscas rápidas)
# max_size=10: permite até 10 conexões simultâneas (pico de uso)
# Se todos os 10 estiverem em uso, a próxima request espera até
# uma conexão ser devolvida ao pool (timeout de 30s por padrão).
#
# Por que min=2?
# - 1 conexão para buscas RAG dos agentes
# - 1 conexão para ingestão de documentos (pode rodar em paralelo)
# Em uso normal, 2 é suficiente. Sob carga, escala até 10.
_supabase_pool: ConnectionPool | None = None

# Pool para o HETZNER (dados operacionais).
# min_size=2: mantém 2 conexões abertas sempre
# max_size=10: permite até 10 conexões simultâneas
#
# Esse pool é usado para:
# - Salvar/carregar chat_memories (toda request de chat)
# - Registrar interaction_logs (toda request de chat)
# - Consultar users/user_sessions (se necessário)
_hetzner_pool: ConnectionPool | None = None


def init_pools() -> None:
    """Inicializa os pools de conexão.
    
    DEVE ser chamada UMA VEZ no startup do servidor (webapp.py).
    Não chamar antes de ter as URLs configuradas no .env.
    
    O que acontece internamente:
    1. ConnectionPool recebe a URL de conexão
    2. Abre min_size conexões imediatamente
    3. Verifica se cada conexão está funcional
    4. Se falhar, levanta exceção (o servidor não sobe com banco offline)
    
    Usar variáveis globais (global _supabase_pool) é a forma padrão
    de gerenciar pools em Python — o pool precisa sobreviver entre
    chamadas de função e ser acessível de qualquer módulo.
    """
    global _supabase_pool, _hetzner_pool
    
    # Inicializa pool do Supabase (vector store)
    if SUPABASE_DB_URL and _supabase_pool is None:
        _supabase_pool = ConnectionPool(
            conninfo=SUPABASE_DB_URL,  # Connection string completa
            min_size=2,                 # Conexões mínimas mantidas abertas
            max_size=10,                # Conexões máximas permitidas
            open=True,                  # Abre as conexões imediatamente
            # kwargs passados para cada conexão individual:
            kwargs={
                "autocommit": True,     # Cada query é um commit automático
                                        # (não precisamos de transações longas)
            },
        )
    
    # Inicializa pool do Hetzner (dados operacionais)
    if HETZNER_DB_URL and _hetzner_pool is None:
        _hetzner_pool = ConnectionPool(
            conninfo=HETZNER_DB_URL,
            min_size=2,
            max_size=10,
            open=True,
            kwargs={
                "autocommit": True,
            },
        )


def close_pools() -> None:
    """Fecha os pools de conexão.
    
    Chamada no shutdown do servidor (webapp.py → evento de shutdown).
    Fecha todas as conexões abertas de forma limpa.
    Sem isso, conexões ficam penduradas no banco até o timeout.
    """
    global _supabase_pool, _hetzner_pool
    
    if _supabase_pool:
        _supabase_pool.close()
        _supabase_pool = None
    
    if _hetzner_pool:
        _hetzner_pool.close()
        _hetzner_pool = None


# ============================================================
# Context managers para obter conexões dos pools
# ============================================================
# 
# Um context manager (usado com "with") garante que a conexão
# é devolvida ao pool mesmo se ocorrer um erro.
#
# Uso:
#   with get_supabase_conn() as conn:
#       with conn.cursor() as cur:
#           cur.execute("SELECT ...")
#           results = cur.fetchall()
#   # Aqui a conexão já foi devolvida ao pool automaticamente
#
# Se não usasse context manager, precisaria de try/finally manual:
#   conn = pool.getconn()
#   try:
#       ...
#   finally:
#       pool.putconn(conn)  # Fácil de esquecer!

@contextmanager
def get_supabase_conn() -> Generator[psycopg.Connection, None, None]:
    """Obtém uma conexão do pool do Supabase.
    
    Uso típico: busca de embeddings, inserção de documentos.
    
    A conexão é emprestada do pool quando entra no "with" e
    devolvida automaticamente quando sai (mesmo com exceção).
    
    Se o pool não foi inicializado (init_pools não chamada),
    levanta erro claro em vez de falhar com NoneType.
    """
    if not _supabase_pool:
        raise RuntimeError(
            "Pool do Supabase não inicializado. "
            "Verifique SUPABASE_DB_URL no .env e se init_pools() foi chamada."
        )
    
    # pool.connection() é um context manager do psycopg_pool:
    # - Pega uma conexão do pool (ou espera se todas estão em uso)
    # - Ao sair do "with", devolve ao pool
    # - Se houve exceção, faz rollback antes de devolver
    with _supabase_pool.connection() as conn:
        yield conn


@contextmanager
def get_hetzner_conn() -> Generator[psycopg.Connection, None, None]:
    """Obtém uma conexão do pool do Postgres Hetzner.
    
    Uso típico: salvar/carregar chat_memories, registrar logs.
    
    Mesma lógica do get_supabase_conn(), mas para o banco
    operacional (Hetzner).
    """
    if not _hetzner_pool:
        raise RuntimeError(
            "Pool do Hetzner não inicializado. "
            "Verifique HETZNER_DB_URL no .env e se init_pools() foi chamada."
        )
    
    with _hetzner_pool.connection() as conn:
        yield conn


# ============================================================
# Função de compatibilidade com o projeto original
# ============================================================

def get_conn(target: str = "supabase") -> psycopg.Connection:
    """Função de compatibilidade com o projeto original.
    
    No projeto do curso, get_conn() retorna uma conexão direta (sem pool).
    Esta versão usa o pool internamente mas expõe a mesma interface.
    
    ATENÇÃO: Prefira usar os context managers (get_supabase_conn, 
    get_hetzner_conn) em código novo. Esta função existe apenas para
    facilitar a migração de código do projeto original que usa get_conn().
    
    Parâmetros:
        target: "supabase" ou "hetzner" — qual banco acessar
    
    Retorna:
        Conexão do pool. IMPORTANTE: o chamador deve fechar/devolver
        a conexão manualmente ou usar um context manager externo.
    """
    if target == "supabase":
        if not _supabase_pool:
            raise RuntimeError("Pool do Supabase não inicializado")
        return _supabase_pool.getconn()
    elif target == "hetzner":
        if not _hetzner_pool:
            raise RuntimeError("Pool do Hetzner não inicializado")
        return _hetzner_pool.getconn()
    else:
        raise ValueError(f"Target inválido: {target}. Use 'supabase' ou 'hetzner'.")
