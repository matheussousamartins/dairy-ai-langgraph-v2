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


def get_calculation_tools() -> list:
    """Retorna as tools de calculo compartilhadas."""
    return [
        calcular_expressao,
        resolver_equacao_linear,
        calcular_diluicao_c1v1,
        calcular_rendimento_percentual,
        listar_formulas_base,
    ]
