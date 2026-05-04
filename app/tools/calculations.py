"""
tools/calculations.py - Ferramentas deterministicas de calculo.

Objetivos:
- Evitar "conta mental" do LLM em formulas da base.
- Permitir calculo por expressao e por equacao linear com incognita.
- Expor um inventario de formulas candidatas extraidas dos documentos.
"""

from __future__ import annotations

import ast
import operator as op
import re
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool


_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

_DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"
_AGENT_DOCS = {
    0: _DOCS_ROOT / "agente-0-base-geral" / "md",
    1: _DOCS_ROOT / "agente-1-queijos" / "md",
    2: _DOCS_ROOT / "agente-2-fermentados" / "md",
    3: _DOCS_ROOT / "agente-3-regulatorios" / "md",
    4: _DOCS_ROOT / "agente-4-qualidade-leite" / "md",
    5: _DOCS_ROOT / "agente-5-defeitos" / "md",
    6: _DOCS_ROOT / "agente-6-formulacao" / "md",
}
_FORMULA_CACHE: Dict[Tuple[Optional[int], int], List[Dict[str, Any]]] = {}


def _to_finite_float(value: Any, field: str) -> float:
    try:
        num = float(value)
    except Exception as exc:
        raise ValueError(f"Valor invalido para '{field}': {value}") from exc
    if not isfinite(num):
        raise ValueError(f"Valor nao finito para '{field}': {value}")
    return num


def _sanitize_var_name(name: str) -> str:
    v = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    if not v:
        raise ValueError("Nome de variavel invalido")
    if v[0].isdigit():
        v = f"v_{v}"
    return v


def _normalize_math_text(text: str) -> str:
    expr = text.strip()
    expr = re.sub(r"(?<=\d),(?=\d)", ".", expr)
    expr = expr.replace("×", "*").replace("·", "*").replace("÷", "/")
    expr = expr.replace("^", "**")
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr


def _prepare_variables(raw: Dict[str, Any]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for k, v in (raw or {}).items():
        clean[_sanitize_var_name(str(k))] = _to_finite_float(v, str(k))
    return clean


def _safe_eval(node: ast.AST, variables: Dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Variavel ausente: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.BinOp):
        fn = _OPS.get(type(node.op))
        if not fn:
            raise ValueError("Operador nao suportado")
        return fn(_safe_eval(node.left, variables), _safe_eval(node.right, variables))
    if isinstance(node, ast.UnaryOp):
        fn = _OPS.get(type(node.op))
        if not fn:
            raise ValueError("Operador nao suportado")
        return fn(_safe_eval(node.operand, variables))
    raise ValueError("Expressao invalida")


def _eval_expression(expression: str, variables: Dict[str, float]) -> float:
    normalized = _normalize_math_text(expression)
    if not normalized or len(normalized) > 500:
        raise ValueError("Expressao invalida")
    try:
        parsed = ast.parse(normalized, mode="eval")
        return _safe_eval(parsed, variables)
    except SyntaxError:
        # Common in OCR/docs where "x" is used as multiplication symbol.
        alt = re.sub(r"(?<=\S)\s+[xX]\s+(?=\S)", " * ", normalized)
        parsed = ast.parse(alt, mode="eval")
        return _safe_eval(parsed, variables)


def _split_equation(equation: str) -> Tuple[str, str]:
    eq = _normalize_math_text(equation)
    if eq.count("=") != 1:
        raise ValueError("Equacao deve conter exatamente um sinal '='")
    lhs, rhs = eq.split("=")
    lhs = lhs.strip()
    rhs = rhs.strip()
    if not lhs or not rhs:
        raise ValueError("Equacao invalida")
    return lhs, rhs


def _equation_residual(equation: str, variables: Dict[str, float]) -> float:
    lhs, rhs = _split_equation(equation)
    return _eval_expression(lhs, variables) - _eval_expression(rhs, variables)


def _is_formula_candidate(line: str) -> bool:
    txt = line.strip()
    if len(txt) < 8 or len(txt) > 260:
        return False
    if "|" in txt:
        return False
    low = txt.lower()
    if low.startswith("onde:"):
        return False
    if "=" not in txt:
        return False
    if re.search(r"\b[CNOHSPK][a-z]?\d", txt):
        return False
    if any(w in low for w in ["solucao", "reagente", "dissolver", "frasco"]):
        return False
    if re.match(
        r"^[a-z0-9_%°\s]+=\s*(volume|massa|fator|normalidade|concentracao|temperatura|tempo)\b",
        low,
    ):
        return False
    if not re.search(r"([*/xX×÷^])|(\s\+\s)", txt):
        return False
    return True


def _extract_formula_candidates_from_file(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return entries

    for idx, line in enumerate(lines, start=1):
        if not _is_formula_candidate(line):
            continue
        raw = line.strip()
        normalized = _normalize_math_text(raw)
        entries.append(
            {
                "source_file": str(path),
                "source_line": idx,
                "raw_formula": raw,
                "normalized_formula": normalized,
            }
        )
    return entries


def _collect_formula_candidates(agent_id: Optional[int], limit: int) -> List[Dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit deve ser >= 1")

    cache_key = (agent_id, limit)
    if cache_key in _FORMULA_CACHE:
        return _FORMULA_CACHE[cache_key]

    if agent_id is None:
        folders: List[Tuple[int, Path]] = list(_AGENT_DOCS.items())
    else:
        folder = _AGENT_DOCS.get(agent_id)
        if not folder:
            raise ValueError("agent_id invalido. Use 0..6")
        folders = [(agent_id, folder)]

    out: List[Dict[str, Any]] = []
    seen = set()

    for aid, folder in folders:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.md")):
            for item in _extract_formula_candidates_from_file(path):
                key = item["normalized_formula"]
                if key in seen:
                    continue
                seen.add(key)
                item["agent_id"] = aid
                out.append(item)
                if len(out) >= limit:
                    _FORMULA_CACHE[cache_key] = out
                    return out

    _FORMULA_CACHE[cache_key] = out
    return out


@tool
def calcular_expressao(
    expression: str,
    variables: Dict[str, float],
    precision: int = 6,
) -> Dict[str, Any]:
    """Calcula uma expressao aritmetica com variaveis.

    Regras:
    - Operadores suportados: +, -, *, /, **, parenteses.
    - Nao permite funcoes, imports, atributos ou chamadas externas.

    Exemplo:
    expression="(acidez_alvo - acidez_atual) * volume_l / fator"
    variables={"acidez_alvo": 18, "acidez_atual": 15, "volume_l": 1000, "fator": 10}
    """
    if not (0 <= precision <= 10):
        raise ValueError("precision deve estar entre 0 e 10")

    clean_vars = _prepare_variables(variables)
    result = _eval_expression(expression, clean_vars)

    return {
        "expression": _normalize_math_text(expression),
        "variables": clean_vars,
        "result": round(result, precision),
        "precision": precision,
    }


@tool
def resolver_equacao_linear(
    equation: str,
    solve_for: str,
    variables: Dict[str, float],
    precision: int = 6,
) -> Dict[str, Any]:
    """Resolve uma equacao linear para uma incognita.

    Exemplo:
    equation="Acidez_D = V * f * 0.9 * 10"
    solve_for="Acidez_D"
    variables={"V": 1.8, "f": 1.0}
    """
    if not equation:
        raise ValueError("equation obrigatoria")
    if not solve_for:
        raise ValueError("solve_for obrigatorio")
    if not (0 <= precision <= 10):
        raise ValueError("precision deve estar entre 0 e 10")

    target = _sanitize_var_name(solve_for)
    clean_vars = _prepare_variables(variables)
    if target in clean_vars:
        raise ValueError(f"Remova '{target}' de variables; ele sera resolvido")

    vars_zero = dict(clean_vars)
    vars_one = dict(clean_vars)
    vars_zero[target] = 0.0
    vars_one[target] = 1.0

    b = _equation_residual(equation, vars_zero)
    a = _equation_residual(equation, vars_one) - b

    if abs(a) < 1e-12:
        raise ValueError("Equacao nao linear ou sem solucao unica para a incognita")

    value = -b / a
    if not isfinite(value):
        raise ValueError("Resultado nao finito")

    check_vars = dict(clean_vars)
    check_vars[target] = value
    residual = _equation_residual(equation, check_vars)

    return {
        "equation": _normalize_math_text(equation),
        "solve_for": target,
        "inputs": clean_vars,
        "result": {target: round(value, precision)},
        "validation": {"residual_abs": abs(round(residual, precision + 2))},
        "precision": precision,
    }


@tool
def calcular_diluicao_c1v1(
    c1: float,
    c2: float,
    v2: float,
    precision: int = 4,
) -> Dict[str, Any]:
    """Calcula V1 na diluicao C1*V1 = C2*V2."""
    c1n = _to_finite_float(c1, "c1")
    c2n = _to_finite_float(c2, "c2")
    v2n = _to_finite_float(v2, "v2")
    if c1n <= 0 or c2n <= 0 or v2n <= 0:
        raise ValueError("c1, c2 e v2 devem ser > 0")
    if c2n > c1n:
        raise ValueError("Para diluicao, c2 nao pode ser maior que c1")
    if not (0 <= precision <= 10):
        raise ValueError("precision deve estar entre 0 e 10")

    v1 = (c2n * v2n) / c1n
    return {
        "formula": "C1*V1 = C2*V2",
        "inputs": {"c1": c1n, "c2": c2n, "v2": v2n},
        "result": {"v1": round(v1, precision)},
        "precision": precision,
    }


@tool
def calcular_rendimento_percentual(
    massa_produto: float,
    massa_materia_prima: float,
    precision: int = 3,
) -> Dict[str, Any]:
    """Calcula rendimento percentual: (massa_produto / massa_materia_prima) * 100."""
    mp = _to_finite_float(massa_produto, "massa_produto")
    mm = _to_finite_float(massa_materia_prima, "massa_materia_prima")
    if mp <= 0 or mm <= 0:
        raise ValueError("massas devem ser > 0")
    if not (0 <= precision <= 10):
        raise ValueError("precision deve estar entre 0 e 10")

    rendimento = (mp / mm) * 100.0
    return {
        "formula": "(massa_produto / massa_materia_prima) * 100",
        "inputs": {"massa_produto": mp, "massa_materia_prima": mm},
        "result": {"rendimento_percentual": round(rendimento, precision)},
        "precision": precision,
    }


@tool
def listar_formulas_base(
    agent_id: Optional[int] = None,
    limit: int = 120,
) -> Dict[str, Any]:
    """Lista formulas candidatas extraidas dos markdowns da base.

    - agent_id: 0..6 ou None para todas.
    - limit: maximo de formulas retornadas.
    """
    if limit < 1 or limit > 1000:
        raise ValueError("limit deve estar entre 1 e 1000")

    items = _collect_formula_candidates(agent_id=agent_id, limit=limit)
    return {
        "agent_id": agent_id,
        "count": len(items),
        "items": items,
    }



# ---------------------------------------------------------------------------
# Catalogo deterministico de formulas do cliente (Principais Formulas.pdf)
# ---------------------------------------------------------------------------

_FC_TABLE = {
    "massa_mole": 1.18,
    "filados": 1.11,
    "continentais": 1.11,
    "duros": 1.09,
}

_FORMULA_CATALOG: Dict[str, Dict[str, Any]] = {
    "van_slyke_original": {
        "nome": "Van Slyke Original",
        "descricao": "Rendimento teorico de queijo em kg/100L de leite.",
        "formula_tex": "Rendimento = [(0.93 * F + C - 0.1) * 1.09] / (1 - W)",
        "parametros": {
            "F": "Gordura do leite (%)",
            "C": "Caseina do leite (%)",
            "W": "Umidade do queijo (decimal, ex: 0.44 para 44%)",
        },
        "unidade": "kg/100L",
    },
    "van_slyke_otimizada": {
        "nome": "Van Slyke Otimizada (Profit)",
        "descricao": "Rendimento teorico com fatores de retencao e correcao por tipo de queijo.",
        "formula_tex": "Rendimento = {[FRF * F] + [PTN * PC * FT] * FC} / (1 - W)",
        "parametros": {
            "FRF": "Fator de retencao de gordura (decimal, ex: 0.90)",
            "F": "Gordura do leite (%)",
            "PTN": "Proteina total do leite (%)",
            "PC": "Caseina como fracao da proteina total (decimal, ex: 0.78)",
            "FT": "Fator de transicao da caseina (decimal, ex: 1.00)",
            "FC": f"Fator de correcao por tipo de queijo — aceita nome ('massa_mole','filados','continentais','duros') ou valor numerico. Tabela: {_FC_TABLE}",
            "W": "Umidade do queijo (decimal)",
        },
        "unidade": "kg/100L",
    },
    "van_slyke_lkg": {
        "nome": "Conversao Van Slyke para L/kg",
        "descricao": "Converte rendimento kg/100L para L/kg de queijo.",
        "formula_tex": "Rendimento_L_kg = (100 / D15) / RVS",
        "parametros": {
            "D15": "Densidade do leite a 15°C (g/mL, ex: 1.032)",
            "RVS": "Resultado Van Slyke em kg/100L",
        },
        "unidade": "L/kg",
    },
    "fleischmann": {
        "nome": "Fleischmann (EST do leite)",
        "descricao": "Extrato Seco Total do leite a partir de gordura e densidade.",
        "formula_tex": "EST = (1.2 * F) + (266.5 * (D15 - 1) / D15) + 0.25",
        "parametros": {
            "F": "Gordura do leite (%)",
            "D15": "Densidade do leite a 15°C (g/mL, ex: 1.032)",
        },
        "unidade": "%",
    },
    "furtado": {
        "nome": "Furtado (EST do leite)",
        "descricao": "Simplificacao de Fleischmann usando graus lactometricos.",
        "formula_tex": "EST = (1.2 * F) + (0.25 * L15) + 0.25",
        "parametros": {
            "F": "Gordura do leite (%)",
            "L15": "Graus lactometricos a 15°C = (D - 1) * 1000",
        },
        "unidade": "%",
    },
    "esd_fleischmann_furtado": {
        "nome": "ESD a partir de Fleischmann/Furtado",
        "descricao": "Extrato Seco Desengordurado = EST - Gordura.",
        "formula_tex": "ESD = RFF - F",
        "parametros": {
            "RFF": "Resultado de Fleischmann ou Furtado (%)",
            "F": "Gordura do leite (%)",
        },
        "unidade": "%",
    },
    "richmond_f1": {
        "nome": "Richmond F1 (EST, L15)",
        "descricao": "EST do leite com leituras a 15°C.",
        "formula_tex": "EST = (L15 / 4) + (1.2 * F) + 0.14",
        "parametros": {
            "L15": "Graus lactometricos a 15°C",
            "F": "Gordura do leite (%)",
        },
        "unidade": "%",
    },
    "richmond_f2": {
        "nome": "Richmond F2 (EST, L20)",
        "descricao": "EST do leite com leituras a 20°C.",
        "formula_tex": "EST = (L20 / 4) + (1.2 * F) + 0.50",
        "parametros": {
            "L20": "Graus lactometricos a 20°C",
            "F": "Gordura do leite (%)",
        },
        "unidade": "%",
    },
    "richmond_f3": {
        "nome": "Richmond F3 (ESD, L15)",
        "descricao": "ESD do leite com leituras a 15°C.",
        "formula_tex": "ESD = (L15 / 4) + (0.2 * F) + 0.14",
        "parametros": {
            "L15": "Graus lactometricos a 15°C",
            "F": "Gordura do leite (%)",
        },
        "unidade": "%",
    },
    "richmond_f4": {
        "nome": "Richmond F4 (ESD, L20)",
        "descricao": "ESD do leite com leituras a 20°C.",
        "formula_tex": "ESD = (L20 / 4) + (0.2 * F) + 0.50",
        "parametros": {
            "L20": "Graus lactometricos a 20°C",
            "F": "Gordura do leite (%)",
        },
        "unidade": "%",
    },
    "densidade_para_graus_lactometricos": {
        "nome": "Densidade para Graus Lactometricos",
        "descricao": "Converte densidade do leite para escala lactometrica.",
        "formula_tex": "L = (D - 1) * 1000",
        "parametros": {
            "D": "Densidade do leite (g/mL, ex: 1.0325)",
        },
        "unidade": "°GL",
    },
    "furtado_soro": {
        "nome": "Furtado Adaptado para Soro",
        "descricao": "EST do soro de queijo.",
        "formula_tex": "EST = (1.2 * F) + (L15 / 4.33) + 0.25",
        "parametros": {
            "F": "Gordura do soro (%)",
            "L15": "Graus lactometricos do soro a 15°C",
        },
        "unidade": "%",
    },
}


def _calc_van_slyke_original(F: float, C: float, W: float) -> float:
    if W >= 1.0 or W < 0:
        raise ValueError("W (umidade) deve ser decimal entre 0 e 1, ex: 0.44 para 44%")
    return ((0.93 * F + C - 0.1) * 1.09) / (1 - W)


def _calc_van_slyke_otimizada(
    FRF: float, F: float, PTN: float, PC: float, FT: float, FC: float, W: float
) -> float:
    if W >= 1.0 or W < 0:
        raise ValueError("W (umidade) deve ser decimal entre 0 e 1")
    return ((FRF * F) + (PTN * PC * FT) * FC) / (1 - W)


def _calc_van_slyke_lkg(D15: float, RVS: float) -> float:
    if D15 <= 0:
        raise ValueError("D15 deve ser > 0")
    if RVS <= 0:
        raise ValueError("RVS deve ser > 0")
    return (100 / D15) / RVS


def _calc_fleischmann(F: float, D15: float) -> float:
    if D15 <= 0:
        raise ValueError("D15 deve ser > 0")
    return (1.2 * F) + (266.5 * (D15 - 1) / D15) + 0.25


def _calc_furtado(F: float, L15: float) -> float:
    return (1.2 * F) + (0.25 * L15) + 0.25


def _calc_esd(RFF: float, F: float) -> float:
    return RFF - F


def _calc_richmond(L: float, F: float, divisor: float, fat_coef: float, intercept: float) -> float:
    return (L / divisor) + (fat_coef * F) + intercept


def _calc_densidade_para_gl(D: float) -> float:
    return (D - 1) * 1000


def _calc_furtado_soro(F: float, L15: float) -> float:
    return (1.2 * F) + (L15 / 4.33) + 0.25


def _resolve_fc(p: Dict[str, Any]) -> float:
    """Aceita FC como float (ex: 1.11) ou como nome de tipo (ex: 'filados')."""
    raw = p.get("FC")
    if raw is None:
        raise ValueError(
            "Parametro 'FC' obrigatorio para van_slyke_otimizada. "
            f"Opcoes por nome: {list(_FC_TABLE.keys())} ou valor numerico direto."
        )
    if isinstance(raw, str):
        key = raw.strip().lower()
        fc = _FC_TABLE.get(key)
        if fc is None:
            raise ValueError(
                f"Tipo de queijo '{raw}' nao reconhecido para FC. "
                f"Opcoes: {list(_FC_TABLE.keys())}"
            )
        return fc
    return _to_finite_float(raw, "FC")


def _require(p: Dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if k not in p]
    if missing:
        raise ValueError(f"Parametros obrigatorios ausentes: {missing}")


_FORMULA_RUNNERS: Dict[str, Any] = {
    "van_slyke_original": lambda p: (
        _require(p, "F", "C", "W") or
        _calc_van_slyke_original(p["F"], p["C"], p["W"])
    ),
    "van_slyke_otimizada": lambda p: (
        _require(p, "FRF", "F", "PTN", "PC", "FT", "W") or
        _calc_van_slyke_otimizada(p["FRF"], p["F"], p["PTN"], p["PC"], p["FT"], _resolve_fc(p), p["W"])
    ),
    "van_slyke_lkg": lambda p: (
        _require(p, "D15", "RVS") or _calc_van_slyke_lkg(p["D15"], p["RVS"])
    ),
    "fleischmann": lambda p: (
        _require(p, "F", "D15") or _calc_fleischmann(p["F"], p["D15"])
    ),
    "furtado": lambda p: (
        _require(p, "F", "L15") or _calc_furtado(p["F"], p["L15"])
    ),
    "esd_fleischmann_furtado": lambda p: (
        _require(p, "RFF", "F") or _calc_esd(p["RFF"], p["F"])
    ),
    "richmond_f1": lambda p: (
        _require(p, "L15", "F") or _calc_richmond(p["L15"], p["F"], 4, 1.2, 0.14)
    ),
    "richmond_f2": lambda p: (
        _require(p, "L20", "F") or _calc_richmond(p["L20"], p["F"], 4, 1.2, 0.50)
    ),
    "richmond_f3": lambda p: (
        _require(p, "L15", "F") or _calc_richmond(p["L15"], p["F"], 4, 0.2, 0.14)
    ),
    "richmond_f4": lambda p: (
        _require(p, "L20", "F") or _calc_richmond(p["L20"], p["F"], 4, 0.2, 0.50)
    ),
    "densidade_para_graus_lactometricos": lambda p: (
        _require(p, "D") or _calc_densidade_para_gl(p["D"])
    ),
    "furtado_soro": lambda p: (
        _require(p, "F", "L15") or _calc_furtado_soro(p["F"], p["L15"])
    ),
}


@tool
def buscar_formula(nome: str) -> Dict[str, Any]:
    """Busca uma formula do catalogo de laticinio pelo nome (slug).

    Retorna a definicao completa: formula, parametros, unidade, descricao.
    Para listar todos os slugs disponíveis, passe nome="listar".

    Slugs disponíveis:
    - van_slyke_original
    - van_slyke_otimizada
    - van_slyke_lkg
    - fleischmann
    - furtado
    - esd_fleischmann_furtado
    - richmond_f1, richmond_f2, richmond_f3, richmond_f4
    - densidade_para_graus_lactometricos
    - furtado_soro
    """
    if nome == "listar":
        return {
            "formulas_disponiveis": [
                {"slug": k, "nome": v["nome"], "descricao": v["descricao"]}
                for k, v in _FORMULA_CATALOG.items()
            ]
        }

    entry = _FORMULA_CATALOG.get(nome)
    if not entry:
        slugs = list(_FORMULA_CATALOG.keys())
        raise ValueError(f"Formula '{nome}' nao encontrada. Slugs validos: {slugs}")

    return dict(entry)


@tool
def calcular_formula_catalogo(
    formula: str,
    parametros: Dict[str, Any],
    precision: int = 4,
) -> Dict[str, Any]:
    """Calcula uma formula do catalogo de laticinio com valores reais.

    Parametros:
    - formula: slug da formula (ex: "van_slyke_original").
      Use buscar_formula("listar") para ver todos os slugs.
    - parametros: dicionario com os valores de entrada. Numeros para a maioria
      dos campos; para van_slyke_otimizada, 'FC' aceita nome do tipo de queijo
      ('massa_mole', 'filados', 'continentais', 'duros') ou valor numerico.
    - precision: casas decimais do resultado (0-10).

    Exemplo — Van Slyke original:
      formula="van_slyke_original"
      parametros={"F": 3.5, "C": 2.65, "W": 0.44}

    Exemplo — Van Slyke Otimizada com nome de tipo:
      formula="van_slyke_otimizada"
      parametros={"FRF": 0.90, "F": 3.5, "PTN": 3.2, "PC": 0.78, "FT": 1.0, "FC": "filados", "W": 0.46}

    Exemplo — Fleischmann:
      formula="fleischmann"
      parametros={"F": 3.5, "D15": 1.032}
    """
    if not (0 <= precision <= 10):
        raise ValueError("precision deve estar entre 0 e 10")

    entry = _FORMULA_CATALOG.get(formula)
    if not entry:
        slugs = list(_FORMULA_CATALOG.keys())
        raise ValueError(f"Formula '{formula}' nao encontrada. Slugs validos: {slugs}")

    runner = _FORMULA_RUNNERS[formula]
    clean_params: Dict[str, Any] = {}
    for k, v in parametros.items():
        if k == "FC" and isinstance(v, str):
            clean_params[k] = v
        else:
            clean_params[k] = _to_finite_float(v, k)

    result = runner(clean_params)

    if not isfinite(result):
        raise ValueError("Resultado nao finito — verifique os parametros")

    return {
        "formula_slug": formula,
        "formula_nome": entry["nome"],
        "formula_tex": entry["formula_tex"],
        "parametros": clean_params,
        "resultado": round(result, precision),
        "unidade": entry["unidade"],
    }


def get_calculation_tools() -> list:
    """Retorna as tools de calculo compartilhadas."""
    return [
        calcular_expressao,
        resolver_equacao_linear,
        calcular_diluicao_c1v1,
        calcular_rendimento_percentual,
        listar_formulas_base,
        buscar_formula,
        calcular_formula_catalogo,
    ]
