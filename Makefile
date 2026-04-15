# ============================================================
# Makefile — Comandos do DairyApp AI (LangGraph)
#
# Adaptado do Makefile do projeto original. Removidos os targets
# de CRM (db_init, test_intents, etc.) e mantidos/adaptados os
# targets de RAG. Adicionados targets para setup e servidor.
# ============================================================

.PHONY: help setup run run_dev test test_fast db_setup \
        rag_phase0 rag_phase1 rag_phase2 rag_phase2_fast \
        rag_phase3 rag_phase3_fast rag_experiments rag_experiments_fast \
        rag_all rag_all_fast

# Detecta Python e prefixos de ambiente por plataforma
ifeq ($(OS),Windows_NT)
PY := $(if $(wildcard .venv/Scripts/python.exe),.venv\\Scripts\\python.exe,python)
PYTHONPATH_RUN := set PYTHONPATH=. &&
RAG_FAST_RUN := set RAG_FAST=1 && set PYTHONPATH=. &&
else
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHONPATH_RUN := PYTHONPATH=.
RAG_FAST_RUN := RAG_FAST=1 PYTHONPATH=.
endif

help:
	@echo ""
	@echo "  DairyApp AI — Comandos disponíveis"
	@echo "  ===================================="
	@echo ""
	@echo "  SETUP"
	@echo "    make setup             Cria venv e instala dependências"
	@echo "    make db_setup          Aplica SQL no Supabase e Hetzner"
	@echo ""
	@echo "  SERVIDOR"
	@echo "    make run               Inicia o servidor FastAPI (porta 8000)"
	@echo "    make run_dev           Inicia com hot reload (desenvolvimento)"
	@echo ""
	@echo "  TESTES GERAIS"
	@echo "    make test              Roda toda a suíte de testes"
	@echo "    make test_fast         Roda testes rápidos (sem API)"
	@echo ""
	@echo "  RAG — FASES DE AVALIAÇÃO"
	@echo "    make rag_phase0        Fase 0: smoke test de ingestão"
	@echo "    make rag_phase1        Fase 1: avaliação de retrieval (vector)"
	@echo "    make rag_phase2        Fase 2: comparação de estratégias (completa)"
	@echo "    make rag_phase2_fast   Fase 2: comparação (10 perguntas)"
	@echo "    make rag_phase3        Fase 3: avaliação E2E do agente (completa)"
	@echo "    make rag_phase3_fast   Fase 3: avaliação E2E (10 perguntas)"
	@echo "    make rag_all           Roda todas as fases (completa)"
	@echo "    make rag_all_fast      Roda todas as fases (fast mode)"
	@echo ""
	@echo "  RAG — EXPERIMENTOS"
	@echo "    make rag_experiments        Runner de experimentos (salva CSV/JSON)"
	@echo "    make rag_experiments_fast   Runner rápido (10 perguntas)"
	@echo "    make rag_experiments_bg     Runner em background (nohup)"
	@echo ""
	@echo "  INGESTÃO"
	@echo "    make ingest DIR=path AGENT=1 TYPE=manual  Ingere documentos de um diretório"
	@echo ""


# ============================================================
# SETUP
# ============================================================

setup:
	@echo "Criando virtual environment..."
	python -m venv .venv
	@echo "Instalando dependências..."
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@echo ""
	@echo "Setup completo! Copie o .env:"
	@echo "  cp .env.example .env"
	@echo "  # Edite .env com suas chaves"

db_setup:
	@echo "Aplicando SQL no Supabase..."
	psql "$(SUPABASE_DB_URL)" < sql/01_kb_schema.sql
	psql "$(SUPABASE_DB_URL)" < sql/02_kb_indexes.sql
	@echo "Aplicando SQL no Hetzner..."
	psql "$(HETZNER_DB_URL)" < sql/03_app_tables.sql
	@echo "Banco de dados configurado!"


# ============================================================
# SERVIDOR
# ============================================================

run:
	$(PYTHONPATH_RUN) $(PY) -m uvicorn app.server.webapp:app --host 0.0.0.0 --port 8000

run_dev:
	$(PYTHONPATH_RUN) $(PY) -m uvicorn app.server.webapp:app --host 0.0.0.0 --port 8000 --reload


# ============================================================
# TESTES GERAIS
# ============================================================

test:
	$(PYTHONPATH_RUN) $(PY) -m pytest -q tests

test_fast:
	$(PYTHONPATH_RUN) $(PY) -m pytest -q -m "not slow and not phase1 and not phase2 and not phase3" tests


# ============================================================
# RAG — FASES
# ============================================================

# Fase 0: Smoke test de ingestão
# Testa: chunking → embeddings → upsert → verificação
# Pré-requisito: OPENAI_API_KEY + SUPABASE_DB_URL
rag_phase0:
	$(PYTHONPATH_RUN) $(PY) -m pytest -q -s -m phase0 tests/integration/rag

# Fase 1: Avaliação de retrieval (busca vetorial por agente)
# Testa: busca retorna chunks relevantes? juiz LLM aprova?
# Pré-requisito: documentos já ingeridos nas tabelas dos agentes
rag_phase1:
	$(PYTHONPATH_RUN) $(PY) -m pytest -q -s -m phase1 tests/integration/rag

# Fase 2: Comparação de estratégias (vector vs text vs hybrid)
# Testa: qual estratégia funciona melhor para cada agente?
rag_phase2:
	$(PYTHONPATH_RUN) $(PY) -m pytest -q -s -m phase2 tests/integration/rag

rag_phase2_fast:
	$(RAG_FAST_RUN) $(PY) -m pytest -q -s -m phase2 tests/integration/rag

# Fase 3: Avaliação E2E (agente completo com prompt + RAG)
# Testa: o agente responde corretamente de ponta a ponta?
rag_phase3:
	$(PYTHONPATH_RUN) $(PY) -m pytest -q -s -m phase3 tests/integration/rag

rag_phase3_fast:
	$(RAG_FAST_RUN) $(PY) -m pytest -q -s -m phase3 tests/integration/rag

# Todas as fases em sequência
rag_all:
	$(PYTHONPATH_RUN) $(PY) -m pytest -q -s tests/integration/rag

rag_all_fast:
	$(RAG_FAST_RUN) $(PY) -m pytest -q -s tests/integration/rag


# ============================================================
# RAG — EXPERIMENTOS (runner com CSV/JSON)
# ============================================================

# Roda todas as combinações do experiments.yaml e salva relatório
rag_experiments:
	$(PYTHONPATH_RUN) $(PY) scripts/rag_experiments_runner.py --outfile experiments

# Versão rápida (10 perguntas por agente)
rag_experiments_fast:
	$(PYTHONPATH_RUN) $(PY) scripts/rag_experiments_runner.py --fast --outfile experiments_fast

# Em background (para rodar combinações completas sem bloquear terminal)
rag_experiments_bg:
	@mkdir -p tests/artifacts/rag/analysis
	nohup env PYTHONPATH=. $(PY) scripts/rag_experiments_runner.py --outfile experiments \
		> tests/artifacts/rag/analysis/experiments.nohup.log 2>&1 & \
		echo $$! > tests/artifacts/rag/analysis/experiments.pid && \
		echo "Iniciado em background (PID $$(cat tests/artifacts/rag/analysis/experiments.pid))"


# ============================================================
# INGESTÃO
# ============================================================

# Ingere documentos .md de um diretório
# Uso: make ingest DIR=./docs/agente-1 AGENT=1 TYPE=manual
ingest:
	@if [ -z "$(DIR)" ] || [ -z "$(AGENT)" ]; then \
		echo "Uso: make ingest DIR=./path AGENT=1 TYPE=manual"; \
		echo "  DIR   = diretório com arquivos .md"; \
		echo "  AGENT = ID do agente (1-6)"; \
		echo "  TYPE  = tipo do documento (manual, legislacao, faq, etc.)"; \
		exit 1; \
	fi
	$(PYTHONPATH_RUN) $(PY) -c "\
from app.rag.ingest import ingest_directory; \
from app.agents.agent_config import get_agent_by_id; \
cfg = get_agent_by_id($(AGENT)); \
result = ingest_directory( \
    base_dir='$(DIR)', \
    table_name=cfg['table_name'], \
    agent_id=$(AGENT), \
    doc_type='$(or $(TYPE),manual)' \
); \
print(result)"
