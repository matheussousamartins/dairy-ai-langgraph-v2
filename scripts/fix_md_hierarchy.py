#!/usr/bin/env python3
"""
fix_md_hierarchy.py — Corrige hierarquia de headings nos markdowns do Agente 1.

Operações realizadas (sem alterar nenhum conteúdo):
  1. Subsections numeradas : ## [**]X.Y título  →  ### [**]X.Y título
  2. Artefatos de tabela   : ## **Label**       →  **Label**  (remove ## )
  3. Anomalias pontuais    : correções explícitas por arquivo

Garantias de segurança:
  - Cria backup .bak antes de qualquer escrita
  - --dry-run: exibe diff sem modificar arquivo
  - Subsections: regex ancorado em ^ + word boundary — nunca toca seções principais
  - Table artifacts: matching por string exata (zero falsos positivos)
  - Anomalias: matching por string exata, substituição cirúrgica
  - Valida que contagem de linhas é idêntica antes e depois
  - Nenhum conteúdo é removido — apenas o prefixo ## é ajustado

Uso:
    python scripts/fix_md_hierarchy.py --dry-run   # preview
    python scripts/fix_md_hierarchy.py              # aplica em todos os arquivos
    python scripts/fix_md_hierarchy.py --file DAIRY_QUEIJOS_DUROS_COMPLETO.md
"""

import re
import shutil
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuração de arquivos
# ---------------------------------------------------------------------------

MD_DIR = Path("docs/agente-1-queijos/md")

ALL_FILES = [
    "DAIRY_QUEIJOS_DUROS_COMPLETO.md",
    "DAIRY_SEMIDUROS_COMPLETO.md",
    "DAIRY_MUSSARELA_COMPLETO.md",
]

# ---------------------------------------------------------------------------
# 1. Subsections numeradas
# ---------------------------------------------------------------------------
# Padrão: linha começa com "## " seguido de opcional "**" e então X.Y
# onde X e Y são 1-2 dígitos (cobre 5.1 a 13.14, mas NÃO 10.000).
# O word boundary \b após os dígitos garante que "32.1 Título" case,
# mas "32.10" (se existir) também é coberto (2+2 dígitos).
#
# Exemplos que casam   : ## 5.1 Título, ## **4.1 Título**, ## 13.14 Título
# Exemplos que não casam: ## **5. Título** (seção principal), ## **10.000 L**

_SUBSECTION_RE = re.compile(
    r"^(## )(\*\*)?(\d{1,2}\.\d{1,2})\b"
)


def _fix_subsection(line: str) -> str | None:
    """Retorna linha corrigida (### ...) ou None se não aplicável."""
    if _SUBSECTION_RE.match(line):
        return "### " + line[3:]  # substitui '## ' por '### '
    return None


# ---------------------------------------------------------------------------
# 2. Artefatos de tabela
# ---------------------------------------------------------------------------
# Strings exatas (sem newline) que são cabeçalhos de coluna de tabela
# mal convertidos como headings Markdown.
# Match por igualdade exata → zero risco de falso positivo.

_TABLE_ARTIFACTS: frozenset[str] = frozenset({
    # ── SEMIDUROS ────────────────────────────────────────────────────────
    "## **Propriedade funcional**",
    "## **Significado tecnológico**",
    "## **Indicador Significado**",
    "## **Microrganismo**",
    "## **Função**",
    "## **Fator controlado na fabricação**",
    "## **Argumento Significado**",
    "## **Objetivo Importância**",
    "## **Modificação Objetivo tecnológico**",
    "## **Modificação**",
    "## **Objetivo tecnológico**",
    "## **Tipo de pasteurização Condição**",
    "## **Fator Efeito**",
    "## **Risco**",
    "## **Explicação**",
    "## **Parâmetro**",
    "## **Uso**",
    "## **Símbolo Significado**",
    "## **Símbolo**",
    "## **Significado**",
    "## **Microrganismos**",
    "## **Características Descrição**",
    "## **Microrganismo Participação aproximada**",
    "## **Efeito**",
    "## **Característica Descrição**",
    "## **Microrganismo Característica**",
    "## **Temperatura de incubação Efeito**",
    "## **Causa**",
    "## **Causa biológica Efeito**",
    "## **Método Utilidade**",
    "## **Reação Produto**",
    "## **Característica Resultado**",
    "## **10.000 L 150 kg a 200 kg**",
    # ── MUSSARELA ────────────────────────────────────────────────────────
    "## **Microrganismo Caracterização tecnológica**",
    "## **Efeito sobre enzimas coagulantes residuais**",
    "## **Efeito Resultado tecnológico**",
    "## **Efeito geral**",
    "## **Defeito**",
    "## **Causa tecnológica**",
    "## **Fator**",
    "## **Efeito potencial**",
    "## **Efeito sobre perdas**",
    "## **Objetivo**",
    "## **Consequência tecnológica**",
    "## **Microrganismo termofílico Sensibilidade ao sal**",
    "## **Propriedade Significado tecnológico**",
    "## **Defeito sensorial Risco associado**",
    "## **Efeito funcional**",
    "## **Região da peça Teor médio de sal**",
    "## **Agente antiaglomerante**",
    "## **Dose ou observação**",
    "## **Mecanismo**",
})


def _fix_table_artifact(line: str) -> str | None:
    """Retorna linha sem o prefixo '## ' ou None se não aplicável."""
    if line.strip() in _TABLE_ARTIFACTS:
        # Remove exatamente '## ' do início (3 chars)
        if line.startswith("## "):
            return line[3:]
    return None


# ---------------------------------------------------------------------------
# 3. Anomalias pontuais (por arquivo)
# ---------------------------------------------------------------------------
# Formato: {nome_arquivo: [(original_exato, corrigido), ...]}
#
# QUEIJOS DUROS linha ~715:
#   Fragmento de listagem de espécies convertido erroneamente como heading.
#   Solução: remove o '## ' sem alterar o conteúdo.
#
# SEMIDUROS linha ~1064:
#   Subscrito ₂ foi parar no início da linha; o CO ficou sem o subscrito.
#   Solução: reposiciona o ₂ e remove o prefixo mal formatado.

_ANOMALIES: dict[str, list[tuple[str, str]]] = {
    "DAIRY_QUEIJOS_DUROS_COMPLETO.md": [
        (
            "## **curvatus** , **Lactobacillus paracasei** subsp. **tolerans** ,"
            " **Lactobacillus brevis** , **Lactobacillus fermentum** e **Pediococcus acidilactici** .",
            "**curvatus** , **Lactobacillus paracasei** subsp. **tolerans** ,"
            " **Lactobacillus brevis** , **Lactobacillus fermentum** e **Pediococcus acidilactici** .",
        ),
    ],
    "DAIRY_SEMIDUROS_COMPLETO.md": [
        (
            "## ₂ **11. Eliminação de odores e CO**",
            "## **11. Eliminação de odores e CO₂**",
        ),
    ],
}


def _fix_anomaly(line: str, filename: str) -> str | None:
    """Retorna linha corrigida se for uma anomalia conhecida, ou None."""
    for original, corrected in _ANOMALIES.get(filename, []):
        if line.strip() == original.strip():
            return corrected
    return None


# ---------------------------------------------------------------------------
# Pipeline de correção
# ---------------------------------------------------------------------------

def _apply_fixes(
    lines: list[str],
    filename: str,
) -> tuple[list[str], list[tuple[int, str, str, str]]]:
    """
    Aplica as três operações a cada linha.

    Retorna:
        (novas_linhas, lista_de_mudanças)
        Cada mudança é (linha_1indexed, tipo, antes, depois).
    """
    result: list[str] = []
    changes: list[tuple[int, str, str, str]] = []

    for i, line in enumerate(lines, start=1):
        raw = line.rstrip("\n")

        fixed = _fix_anomaly(raw, filename)
        kind = "anomalia"

        if fixed is None:
            fixed = _fix_table_artifact(raw)
            kind = "table_artifact"

        if fixed is None:
            fixed = _fix_subsection(raw)
            kind = "subsection"

        if fixed is not None and fixed != raw:
            changes.append((i, kind, raw, fixed))
            result.append(fixed + "\n")
        else:
            result.append(line)

    return result, changes


def process_file(
    filepath: Path,
    dry_run: bool,
) -> dict:
    """Processa um arquivo e retorna estatísticas."""
    original_text = filepath.read_text(encoding="utf-8")
    lines = original_text.splitlines(keepends=True)
    original_count = len(lines)

    new_lines, changes = _apply_fixes(lines, filepath.name)

    # Validação de integridade: contagem de linhas deve ser igual
    assert len(new_lines) == original_count, (
        f"ERRO: contagem de linhas divergiu em {filepath.name} "
        f"({original_count} → {len(new_lines)}). Abortando."
    )

    stats = {
        "file": filepath.name,
        "total_lines": original_count,
        "changes": len(changes),
        "subsections": sum(1 for _, k, _, _ in changes if k == "subsection"),
        "table_artifacts": sum(1 for _, k, _, _ in changes if k == "table_artifact"),
        "anomalias": sum(1 for _, k, _, _ in changes if k == "anomalia"),
        "dry_run": dry_run,
    }

    _print_report(stats, changes)

    if not dry_run and changes:
        backup = filepath.with_suffix(".md.bak")
        shutil.copy2(filepath, backup)
        filepath.write_text("".join(new_lines), encoding="utf-8")
        print(f"  Backup  : {backup.name}")
        print(f"  Salvo   : {filepath.name}")

    return stats


def _print_report(
    stats: dict,
    changes: list[tuple[int, str, str, str]],
) -> None:
    prefix = "[DRY-RUN] " if stats["dry_run"] else ""
    print(f"\n{'─' * 60}")
    print(f"{prefix}{stats['file']}")
    print(f"  Linhas total     : {stats['total_lines']}")
    print(f"  Subsections (###): {stats['subsections']}")
    print(f"  Table artifacts  : {stats['table_artifacts']}")
    print(f"  Anomalias        : {stats['anomalias']}")
    print(f"  Total de mudanças: {stats['changes']}")

    if changes:
        print()
        for lineno, kind, before, after in changes:
            tag = {"subsection": "SUB", "table_artifact": "TBL", "anomalia": "ANO"}[kind]
            print(f"  [{tag}] L{lineno:>5}  {before[:70]}")
            print(f"           →  {after[:70]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corrige hierarquia de headings nos markdowns do Agente 1"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exibe mudanças sem modificar os arquivos",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Processa apenas este arquivo (ex: DAIRY_QUEIJOS_DUROS_COMPLETO.md)",
    )
    args = parser.parse_args()

    targets = [args.file] if args.file else ALL_FILES

    print("=" * 60)
    print("fix_md_hierarchy.py")
    print("Modo: DRY-RUN (sem escrita)" if args.dry_run else "Modo: APLICAR")
    print("=" * 60)

    totals = {"changes": 0, "subsections": 0, "table_artifacts": 0, "anomalias": 0}

    for fname in targets:
        fpath = MD_DIR / fname
        if not fpath.exists():
            print(f"\nERRO: arquivo não encontrado: {fpath}")
            continue
        stats = process_file(fpath, dry_run=args.dry_run)
        for key in totals:
            totals[key] += stats[key]

    print(f"\n{'=' * 60}")
    print("RESUMO GERAL")
    print(f"  Subsections (##→###) : {totals['subsections']}")
    print(f"  Table artifacts      : {totals['table_artifacts']}")
    print(f"  Anomalias            : {totals['anomalias']}")
    print(f"  Total de mudanças    : {totals['changes']}")
    if args.dry_run:
        print("\n(dry-run: nenhum arquivo foi modificado)")
    print("=" * 60)


if __name__ == "__main__":
    main()
