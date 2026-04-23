"""
Conversão de PDF para Markdown usando GPT-4o Vision.

Renderiza cada página do PDF como imagem (via PyMuPDF) e envia ao GPT-4o
para extração fiel do conteúdo: texto, tabelas e gráficos com valores.

Ideal para documentos com infográficos, gráficos técnicos e diagramas onde
extratores de texto tradicionais (pdfplumber, markitdown) perdem informação.

Uso:
  # Converte todos os PDFs do agente 1
  python scripts/pdf_to_md_vision.py --agent 1

  # Converte um PDF específico
  python scripts/pdf_to_md_vision.py --file docs/agente-1-queijos/pdf/arquivo.pdf --out docs/agente-1-queijos/md/

  # Pré-visualiza sem salvar
  python scripts/pdf_to_md_vision.py --file docs/agente-1-queijos/pdf/arquivo.pdf --dry-run
"""

import argparse
import base64
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError

# Garante output UTF-8 no terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

DPI_SCALE = 2.0          # 2x = ~144 dpi — boa resolução para OCR + compacto
MAX_TOKENS_PER_PAGE = 4096
GPT_MODEL = "gpt-4o"
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 5.0   # segundos (dobra a cada tentativa)
INTER_PAGE_DELAY = 1.0   # segundos entre chamadas para evitar rate limit

# Cabeçalhos/rodapés repetitivos que aparecem em todas as páginas da Ha-La Biotec
# São ignorados após a primeira página para evitar redundância no markdown final
REPEATED_HEADER_PATTERNS = [
    r"^(#+\s*)?Ha-La\s+Biotec\s*$",
    r"^NÚMERO\s*\d+\s*\|",
    r"^www\.halabiotec\.com",
    r"^\d+\s+Ha-La\s+Biotec\s+\d+",  # "2 Ha-La Biotec 169"
]

SYSTEM_PROMPT = """Você é um especialista em extração de conteúdo técnico de documentos PDF para uso em sistemas RAG (Recuperação Aumentada por Geração).

Sua tarefa é transcrever fielmente o conteúdo de cada página para Markdown bem estruturado.

REGRAS ABSOLUTAS:
1. Preserve 100% do conteúdo informativo original — não omita nada técnico
2. Corrija apenas problemas de espaçamento entre palavras coladas (sem inventar conteúdo)
3. Não adicione comentários, explicações ou contexto próprios
4. Não repita informações que claramente são cabeçalho/rodapé decorativo (logo, número da edição)

FORMATAÇÃO DE TEXTO:
- Use headers Markdown (##, ###, ####) respeitando a hierarquia visual da página
- Preserve negritos e itálicos quando visíveis
- Listas com marcadores para itens enumerados

FORMATAÇÃO DE TABELAS:
- Converta TODA tabela em tabela Markdown com | col | col |
- Preserve todos os valores numéricos exatamente como estão
- Inclua linha de separação (|---|---|) obrigatoriamente

FORMATAÇÃO DE GRÁFICOS E INFOGRÁFICOS:
Use SEMPRE este formato para gráficos:

> **[Gráfico: título exato do gráfico]**
> Tipo: [linha/barra/dispersão/pizza/boxplot/outro]
> Eixo X: [label e unidade] | Eixo Y: [label e unidade]
> Séries/Dados:
> - [Série 1]: [valores ou tendência descrita]
> - [Série 2]: [valores ou tendência descrita]
> Observações: [conclusão visual principal se houver legenda ou anotação no gráfico]

FORMATAÇÃO DE IMAGENS TÉCNICAS:
Para fotos, microscopia, diagramas de processo, use:

> **[Imagem: descrição curta]**
> [Descrição do que a imagem mostra em 1-3 frases técnicas, incluindo o que é relevante para um especialista em laticínios/queijos]

ELEMENTOS A IGNORAR (puramente decorativos):
- Bordas e linhas decorativas
- Ícones de redes sociais
- Números de página isolados sem contexto"""

PAGE_USER_PROMPT = """Extraia o conteúdo desta página seguindo rigorosamente as instruções do sistema.

Esta é a página {page_num} de {total_pages} do documento "{doc_name}".
{continuation_note}

Retorne apenas o conteúdo Markdown extraído, sem blocos de código envolvendo o resultado."""


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def render_page_to_b64(page: fitz.Page, scale: float = DPI_SCALE) -> str:
    """Renderiza página PDF como PNG base64."""
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode()


def is_repeated_header(line: str) -> bool:
    """Verifica se uma linha é cabeçalho repetitivo da Ha-La Biotec."""
    for pattern in REPEATED_HEADER_PATTERNS:
        if re.match(pattern, line.strip(), re.IGNORECASE):
            return True
    return False


def clean_page_content(content: str, is_first_page: bool) -> str:
    """
    Remove cabeçalhos repetitivos de páginas subsequentes e
    normaliza espaçamento excessivo.
    """
    if is_first_page:
        return content.strip()

    lines = content.split("\n")
    cleaned = []
    skip_next_blank = False

    for line in lines:
        if is_repeated_header(line):
            skip_next_blank = True
            continue
        if skip_next_blank and line.strip() == "":
            skip_next_blank = False
            continue
        skip_next_blank = False
        cleaned.append(line)

    # Remove múltiplas linhas em branco consecutivas (máx 2)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


def extract_page_with_vision(
    client: OpenAI,
    page_b64: str,
    page_num: int,
    total_pages: int,
    doc_name: str,
    is_first_page: bool,
) -> str:
    """Extrai conteúdo de uma página via GPT-4o vision com retry."""
    continuation_note = (
        ""
        if is_first_page
        else "Esta é uma página de continuação — não reintroduza o título do documento."
    )

    user_content = PAGE_USER_PROMPT.format(
        page_num=page_num,
        total_pages=total_pages,
        doc_name=doc_name,
        continuation_note=continuation_note,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{page_b64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]

    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=MAX_TOKENS_PER_PAGE,
                messages=messages,
            )
            content = resp.choices[0].message.content or ""
            # Remove wrapper de bloco de código se o modelo incluiu
            content = re.sub(r"^```(?:markdown)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
            return content.strip()

        except RateLimitError:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    [wait] Rate limit — aguardando {delay:.0f}s...", flush=True)
            time.sleep(delay)
        except APIError as e:
            if attempt < RETRY_ATTEMPTS - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"    [warn]  API error ({e}) — retry em {delay:.0f}s...", flush=True)
                time.sleep(delay)
            else:
                raise

    raise RuntimeError(f"Falhou após {RETRY_ATTEMPTS} tentativas na página {page_num}")


def build_doc_header(pdf_path: Path, total_pages: int) -> str:
    """Monta cabeçalho padronizado do documento markdown."""
    name = pdf_path.stem
    # Extrai número da edição se presente (ex: "Edicao_152", "Edição_138")
    match = re.search(r"[Ee]di(?:cao|ção|cão|cao)_?(\d+)", name)
    edition = f"Edição {match.group(1)}" if match else name

    return f"<!-- source: {pdf_path.name} | pages: {total_pages} | converted: vision/gpt-4o -->\n\n"


def pdf_to_markdown(
    pdf_path: Path,
    output_dir: Path,
    client: OpenAI,
    dry_run: bool = False,
) -> Optional[Path]:
    """
    Converte um PDF para Markdown usando GPT-4o vision.

    Args:
        pdf_path: Caminho do PDF de entrada
        output_dir: Diretório de saída para o .md
        client: Cliente OpenAI configurado
        dry_run: Se True, imprime resultado sem salvar

    Returns:
        Path do arquivo .md gerado, ou None em dry_run
    """
    doc_name = pdf_path.stem
    print(f"\n{'='*60}", flush=True)
    print(f"[PDF] {pdf_path.name}", flush=True)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"   {total_pages} páginas | renderizando em {DPI_SCALE}x...", flush=True)

    pages_md: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, page in enumerate(doc):
        page_num = i + 1
        is_first = i == 0
        print(f"   Página {page_num}/{total_pages}...", end=" ", flush=True)

        page_b64 = render_page_to_b64(page)
        size_kb = len(page_b64) * 3 // 4 // 1024
        print(f"({size_kb}KB PNG)", end=" ", flush=True)

        content = extract_page_with_vision(
            client=client,
            page_b64=page_b64,
            page_num=page_num,
            total_pages=total_pages,
            doc_name=doc_name,
            is_first_page=is_first,
        )
        content = clean_page_content(content, is_first_page=is_first)
        pages_md.append(content)
        print("[ok]", flush=True)

        if page_num < total_pages:
            time.sleep(INTER_PAGE_DELAY)

    doc.close()

    # Monta documento final
    header = build_doc_header(pdf_path, total_pages)
    full_md = header + "\n\n".join(pages_md)

    # Normaliza espaçamento final
    full_md = re.sub(r"\n{4,}", "\n\n\n", full_md)
    full_md = full_md.strip() + "\n"

    if dry_run:
        print("\n--- DRY RUN OUTPUT ---\n")
        print(full_md[:3000])
        if len(full_md) > 3000:
            print(f"\n... ({len(full_md) - 3000} chars omitidos)")
        return None

    # Nome do arquivo de saída: sanitiza o nome do PDF
    safe_name = re.sub(r"[^\w\-_. ]", "_", pdf_path.stem)
    safe_name = re.sub(r"\s+", "_", safe_name)
    safe_name = re.sub(r"_+", "_", safe_name).strip("_")
    out_path = output_dir / f"{safe_name}.md"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_md, encoding="utf-8")

    size_kb = len(full_md.encode("utf-8")) // 1024
    print(f"   [OK] Salvo: {out_path.name} ({size_kb}KB, {len(pages_md)} páginas)", flush=True)

    return out_path


# ---------------------------------------------------------------------------
# Descoberta de PDFs por agente
# ---------------------------------------------------------------------------

AGENT_PDF_DIRS = {
    1: [
        Path("docs/agente-1-queijos/pdf"),
        Path("docs/agente-1-queijos/pdf/ha_la_defeitos_em_queijos"),
    ],
    2: [Path("docs/agente-2-fermentados/pdf")],
    3: [Path("docs/agente-3-regulatorios/pdf")],
    4: [Path("docs/agente-4-qualidade-leite/pdf")],
    5: [Path("docs/agente-5-defeitos/pdf")],
    6: [Path("docs/agente-6-formulacao/pdf")],
}

AGENT_MD_DIRS = {
    1: Path("docs/agente-1-queijos/md"),
    2: Path("docs/agente-2-fermentados/md"),
    3: Path("docs/agente-3-regulatorios/md"),
    4: Path("docs/agente-4-qualidade-leite/md"),
    5: Path("docs/agente-5-defeitos/md"),
    6: Path("docs/agente-6-formulacao/md"),
}


def find_pdfs_for_agent(agent_id: int) -> list[Path]:
    """Retorna todos os PDFs do agente, excluindo .gitkeep."""
    dirs = AGENT_PDF_DIRS.get(agent_id, [])
    pdfs = []
    for d in dirs:
        if d.exists():
            pdfs.extend(sorted(d.glob("*.pdf")))
    return pdfs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte PDFs para Markdown usando GPT-4o Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--agent",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="ID do agente (converte todos os PDFs do agente)",
    )
    group.add_argument(
        "--file",
        type=Path,
        help="Caminho de um PDF específico",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Diretório de saída (obrigatório com --file, ignorado com --agent)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Imprime resultado no terminal sem salvar arquivo",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Pula PDFs que já têm .md correspondente (padrão: ativo)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Reconverte mesmo que .md já exista",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERR] OPENAI_API_KEY não encontrada no .env", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Monta lista de trabalho
    if args.file:
        if not args.file.exists():
            print(f"[ERR] Arquivo não encontrado: {args.file}", file=sys.stderr)
            sys.exit(1)
        out_dir = args.out
        if not out_dir:
            print("[ERR] --out é obrigatório quando usando --file", file=sys.stderr)
            sys.exit(1)
        work = [(args.file, out_dir)]
    else:
        pdfs = find_pdfs_for_agent(args.agent)
        if not pdfs:
            print(f"[ERR] Nenhum PDF encontrado para agente {args.agent}", file=sys.stderr)
            sys.exit(1)
        out_dir = AGENT_MD_DIRS[args.agent]
        work = [(p, out_dir) for p in pdfs]

    print(f"[INFO] {len(work)} PDF(s) encontrado(s)")

    # Filtra já convertidos
    if args.skip_existing and not args.dry_run:
        filtered = []
        for pdf_path, out_d in work:
            safe_name = re.sub(r"[^\w\-_. ]", "_", pdf_path.stem)
            safe_name = re.sub(r"\s+", "_", safe_name)
            safe_name = re.sub(r"_+", "_", safe_name).strip("_")
            md_path = out_d / f"{safe_name}.md"
            if md_path.exists():
                print(f"   [SKIP]  Pulando (já existe): {md_path.name}")
            else:
                filtered.append((pdf_path, out_d))
        work = filtered

    if not work:
        print("[OK] Nada a converter.")
        return

    print(f"[RUN] Convertendo {len(work)} PDF(s) com GPT-4o vision...\n")

    converted = []
    failed = []

    for pdf_path, out_d in work:
        try:
            result = pdf_to_markdown(
                pdf_path=pdf_path,
                output_dir=out_d,
                client=client,
                dry_run=args.dry_run,
            )
            if result:
                converted.append(result)
        except Exception as e:
            print(f"\n[ERR] Erro em {pdf_path.name}: {e}", file=sys.stderr)
            failed.append(pdf_path)

    # Resumo
    print(f"\n{'='*60}")
    print(f"[OK] Convertidos: {len(converted)}")
    if failed:
        print(f"[ERR] Falhas: {len(failed)}")
        for f in failed:
            print(f"   - {f.name}")
    if converted:
        print("\nArquivos gerados:")
        for p in converted:
            size_kb = p.stat().st_size // 1024
            print(f"   {p}  ({size_kb}KB)")


if __name__ == "__main__":
    main()
