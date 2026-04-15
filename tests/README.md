# Test Suite Layout

Estrutura recomendada para manter testes organizados e previsíveis:

- `integration/rag/`: testes de integração por fase (`phase0`..`phase3`)
- `fixtures/rag/`: datasets e arquivos de configuração de experimento
- `artifacts/rag/analysis/`: saídas geradas por runners (CSV/JSON/logs)
- `conftest.py`: fixtures compartilhadas da suíte

Comandos úteis:

- `python -m pytest -q tests`
- `python -m pytest -q -m phase0 tests/integration/rag`
- `python -m pytest -q -m phase1 tests/integration/rag`
- `python -m pytest -q -m phase2 tests/integration/rag`
- `python -m pytest -q -m phase3 tests/integration/rag`
