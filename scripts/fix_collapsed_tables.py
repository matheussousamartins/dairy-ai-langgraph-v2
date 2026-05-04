#!/usr/bin/env python3
"""
Fix collapsed tables in markdown files converted from PDF.
Uses OpenAI API to reconstruct proper markdown tables.

Safety guarantees:
- Creates a .bak backup before any modification
- Never touches lines that don't match the collapsed-table pattern
- Validates LLM output before replacing (must contain | pipe chars)
- If validation fails, keeps original line untouched
- Logs every change and every skip with reason
"""

import re
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MD_FILE = Path("docs/agente-1-queijos/md/DAIRY - MUSSARELA COMPLETO.md")
BACKUP_FILE = MD_FILE.with_suffix(".md.bak")
LOG_FILE = Path("docs/agente-1-queijos/md/fix_tables_mussarela.log")

# Matches lines like: **Column1 Header Column2 Header** followed by data content
# These are tables where the PDF two-column layout was flattened into one line
COLLAPSED_PATTERN = re.compile(
    r'^\*\*[A-ZÁÉÍÓÚÃÕÂÊÔÜ][^\*]{3,}\*\* [A-ZÁÉÍÓÚÃÕÂÊÔÜ]'
)

SYSTEM_PROMPT = """\
You are a markdown table reconstruction specialist working on a dairy technology document in Portuguese.

Your ONLY task: convert a single collapsed table line back into a proper markdown table.

Rules (non-negotiable):
1. Do NOT change, add, or remove any words, numbers, percentages, temperatures, units, or scientific terms.
2. Only add markdown table structure: header row, separator row (---|---), and data rows.
3. Return ONLY the markdown table — no explanation, no preamble, no trailing text.
4. Preserve all formatting inside cells: **bold**, _italic_, °C, %, etc.
5. If the line contains bold text like **Col1 Col2** at the start, those are the column headers.
6. Each data pair on the line becomes one row.
7. If you cannot determine the structure with confidence, return the original line unchanged."""


def is_valid_table(text: str) -> bool:
    """Verify the LLM output looks like a real markdown table."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    has_pipe = any("|" in l for l in lines)
    has_separator = any(re.match(r'\|[-| :]+\|', l) for l in lines)
    return has_pipe and has_separator


def reconstruct_table(client: OpenAI, collapsed: str, before: str, after: str) -> tuple[str, bool]:
    """
    Ask the LLM to reconstruct a collapsed table.
    Returns (result_text, was_changed).
    """
    user_msg = (
        "Context before the collapsed table:\n"
        f"{before}\n\n"
        "COLLAPSED TABLE LINE — reconstruct this into markdown table format:\n"
        f"{collapsed}\n\n"
        "Context after:\n"
        f"{after}\n\n"
        "Return only the markdown table. Do not change any content."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    result = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wrapped in ```markdown ... ```
    result = re.sub(r'^```(?:markdown)?\s*', '', result)
    result = re.sub(r'\s*```$', '', result).strip()

    return result, True


def main():
    print("=" * 60)
    print("fix_collapsed_tables.py")
    print("=" * 60)

    if not MD_FILE.exists():
        print(f"ERRO: Arquivo não encontrado: {MD_FILE}")
        return

    # Backup
    shutil.copy2(MD_FILE, BACKUP_FILE)
    print(f"Backup criado: {BACKUP_FILE}\n")

    lines = MD_FILE.read_text(encoding="utf-8").splitlines()
    original_line_count = len(lines)

    # Detect collapsed table lines (original indices)
    collapsed_indices = [
        i for i, line in enumerate(lines)
        if COLLAPSED_PATTERN.match(line.strip())
    ]
    print(f"Tabelas colapsadas detectadas: {len(collapsed_indices)}")
    print(f"Total de linhas no arquivo:    {original_line_count}\n")

    client = OpenAI()
    log_entries = []
    new_lines = lines.copy()
    offset = 0  # Accumulates as we insert multi-line tables
    fixed = 0
    skipped_invalid = 0
    skipped_error = 0

    for seq, orig_idx in enumerate(collapsed_indices, 1):
        actual_idx = orig_idx + offset
        collapsed_line = new_lines[actual_idx]

        # Context window: 4 lines before and 4 after
        before_start = max(0, actual_idx - 4)
        after_end = min(len(new_lines), actual_idx + 5)
        before = "\n".join(new_lines[before_start:actual_idx])
        after = "\n".join(new_lines[actual_idx + 1:after_end])

        print(f"[{seq:03d}/{len(collapsed_indices)}] Linha orig {orig_idx + 1} ...", end=" ", flush=True)

        try:
            result, _ = reconstruct_table(client, collapsed_line, before, after)

            if not is_valid_table(result):
                # LLM output didn't produce a real table — keep original
                print("IGNORADO (output inválido)")
                log_entries.append(
                    f"SKIP_INVALID | orig_line={orig_idx + 1} | collapsed={collapsed_line[:80]}"
                )
                skipped_invalid += 1
            else:
                result_lines = result.splitlines()
                new_lines[actual_idx: actual_idx + 1] = result_lines
                offset += len(result_lines) - 1
                fixed += 1
                print(f"OK ({len(result_lines)} linhas)")
                log_entries.append(
                    f"FIXED | orig_line={orig_idx + 1} | out_lines={len(result_lines)} | collapsed={collapsed_line[:80]}"
                )

        except Exception as exc:
            print(f"ERRO: {exc}")
            log_entries.append(
                f"ERROR | orig_line={orig_idx + 1} | err={exc} | collapsed={collapsed_line[:80]}"
            )
            skipped_error += 1

        # Be polite to the API
        time.sleep(0.25)

    # Write corrected file
    MD_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # Write log
    LOG_FILE.write_text("\n".join(log_entries) + "\n", encoding="utf-8")

    new_line_count = len(new_lines)
    print("\n" + "=" * 60)
    print(f"Corrigidas:          {fixed}")
    print(f"Ignoradas (inválido):{skipped_invalid}")
    print(f"Erros:               {skipped_error}")
    print(f"Linhas antes:        {original_line_count}")
    print(f"Linhas depois:       {new_line_count}")
    print(f"Arquivo salvo:       {MD_FILE}")
    print(f"Log:                 {LOG_FILE}")
    print(f"Backup:              {BACKUP_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
