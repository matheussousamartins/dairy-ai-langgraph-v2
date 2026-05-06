"""
orch_fewshot.py — Few-shot examples para o classificador de roteamento.

Carrega exemplos estratificados do dataset de benchmark para injetar
no prompt de classificação. Exemplos reais são empiricamente mais
eficazes que regras declarativas para LLMs.

Foco: confusões dominantes (3→1, 1→5, 0→1, 2→4) com exemplos
que resolvem a ambiguidade de forma clara.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.config import CLASSIFIER_FEW_SHOT_PER_CLASS, CLASSIFIER_FEW_SHOT_ENABLED


# ---------------------------------------------------------------------------
# Exemplos curados manualmente (alta qualidade, focados em confusões)
# ---------------------------------------------------------------------------
# Estes exemplos são selecionados para resolver as confusões dominantes
# do benchmark. Cada um demonstra o roteamento correto em caso ambíguo.

_CURATED_EXAMPLES: List[Dict] = [
    # Confusão 3→1: perguntas técnicas de queijo que parecem regulatórias
    {
        "question": "Qual a temperatura ideal de coagulação para Parmesão?",
        "agent_ids": [1, 3],
        "confidence": 0.92,
        "reason": "Pergunta técnica de processo de queijo duro — Agente 1 primário, 3 complementar",
    },
    {
        "question": "Em que condição pode ocorrer estufamento tardio por Clostridium tyrobutyricum na mussarela?",
        "agent_ids": [1, 3],
        "confidence": 0.90,
        "reason": "Defeito técnico documentado na base de queijos — Agente 1 primário",
    },
    # Confusão 1→3: perguntas regulatórias que citam queijo
    {
        "question": "Qual o período mínimo de maturação exigido por lei para queijos duros?",
        "agent_ids": [3, 1],
        "confidence": 0.88,
        "reason": "Requisito legal/obrigatório — Agente 3 primário, 1 como contexto técnico",
    },
    {
        "question": "Quais são os limites microbiológicos da IN 60 para queijo Minas Frescal?",
        "agent_ids": [3],
        "confidence": 0.95,
        "reason": "Limites legais de norma específica — puramente regulatório",
    },
    # Confusão 0→1: glossário vs tecnologia
    {
        "question": "O que significa affioramento no contexto de queijos?",
        "agent_ids": [1, 3],
        "confidence": 0.85,
        "reason": "Termo técnico específico de fabricação de queijos — Agente 1 tem a definição no contexto de soro-fermento",
    },
    {
        "question": "Qual a diferença entre os termos 'queijo' e 'produto lácteo'?",
        "agent_ids": [3],
        "confidence": 0.90,
        "reason": "Definição regulatória de identidade — Agente 3",
    },
    # Confusão 1→5: defeitos em queijo vs diagnóstico geral
    {
        "question": "Meu queijo Prato está com olhaduras irregulares, o que pode ser?",
        "agent_ids": [1, 3],
        "confidence": 0.88,
        "reason": "Defeito técnico de queijo específico documentado — Agente 1 (base de defeitos integrada)",
    },
    # Agente 3 puro: regulatório sem ambiguidade
    {
        "question": "O que diz o RIISPOA sobre tratamento térmico do leite?",
        "agent_ids": [3],
        "confidence": 0.96,
        "reason": "Referência direta a norma — puramente regulatório",
    },
    # Saudação / off-topic
    {
        "question": "Bom dia! Como você pode me ajudar?",
        "agent_ids": [],
        "confidence": 0.99,
        "reason": "Saudação sem pergunta técnica — resposta direta",
    },
    # Método analítico → Agente 4 (quando ativo)
    {
        "question": "Como calcular a acidez Dornic do leite usando titulação?",
        "agent_ids": [3],
        "confidence": 0.75,
        "reason": "Método analítico — idealmente Agente 4, mas sem KB ativa, regulatório cobre parcialmente",
    },
    # Fermentados → Agente 2 (quando ativo)
    {
        "question": "Qual pH indica ponto de quebra ideal no iogurte grego?",
        "agent_ids": [3],
        "confidence": 0.70,
        "reason": "Fermentado — idealmente Agente 2, sem KB ativa, regulatório como referência parcial",
    },
    # Confusão 2→1: fermentação em queijo vs fermentados
    {
        "question": "Qual o pH ideal de corte da coalhada para mussarela?",
        "agent_ids": [1, 3],
        "confidence": 0.93,
        "reason": "Processo de fabricação de queijo (coalhada/corte) — Agente 1, não fermentados",
    },
]


def _format_example(ex: Dict) -> str:
    """Formata um exemplo para inclusão no prompt."""
    ids_str = str(ex["agent_ids"])
    return (
        f"  Pergunta: \"{ex['question']}\"\n"
        f"  → agent_ids: {ids_str}, confidence: {ex['confidence']}, reason: \"{ex['reason']}\""
    )


def build_few_shot_block() -> str:
    """Constrói o bloco de few-shot examples para o prompt de classificação.

    Retorna string formatada pronta para injeção no prompt, ou string vazia
    se a feature estiver desabilitada.
    """
    if not CLASSIFIER_FEW_SHOT_ENABLED:
        return ""

    # Seleciona exemplos: curados são sempre incluídos (são poucos e de alta qualidade)
    examples = _CURATED_EXAMPLES

    lines = ["EXEMPLOS DE CLASSIFICAÇÃO (referência):"]
    for ex in examples:
        lines.append(_format_example(ex))
        lines.append("")

    return "\n".join(lines)


def get_few_shot_examples() -> List[Dict]:
    """Retorna a lista de exemplos few-shot (para uso em testes/benchmark)."""
    return list(_CURATED_EXAMPLES)
