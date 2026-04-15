"""
scripts/rag_experiments_runner.py — Runner de experimentos RAG

Executa todas as combinações de parâmetros definidas em
experiments.yaml contra o dataset rag_queries.yaml e gera
um relatório CSV com os resultados.

Adaptado dos scripts/rag_phase2_runner.py e rag_phase3_runner.py
do projeto original, combinados em um único runner.

Uso:
  make rag_experiments              (todas as perguntas)
  make rag_experiments_fast         (10 perguntas)
  
  Ou direto:
  PYTHONPATH=. python scripts/rag_experiments_runner.py --fast --outfile results

Saída:
  tests/artifacts/rag/analysis/results.csv    — resultados em CSV
  tests/artifacts/rag/analysis/results.json   — resultados em JSON

Formato do CSV:
  experiment,agent_id,total,hits,hit_rate,llm_yes,llm_rate
  fixed_vector,1,8,6,0.75,5,0.63
  fixed_hybrid_rrf,1,8,7,0.88,6,0.75
  md_vector,3,6,4,0.67,3,0.50
  ...
"""

import os
import sys
import json
import csv
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime

import yaml
import psycopg
from dotenv import load_dotenv


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def load_experiments() -> list:
    """Carrega combinações de experiments.yaml (ou template se não existir)."""
    base = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rag"
    exp_file = base / "experiments.yaml"
    if not exp_file.exists():
        exp_file = base / "experiments.template.yaml"
    if not exp_file.exists():
        print("ERRO: experiments.yaml não encontrado")
        sys.exit(1)
    data = yaml.safe_load(exp_file.read_text(encoding="utf-8")) or {}
    return data.get("experiments") or []


def load_dataset() -> list:
    """Carrega o dataset de perguntas."""
    ds_file = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rag" / "rag_queries.yaml"
    if not ds_file.exists():
        print("ERRO: rag_queries.yaml não encontrado")
        sys.exit(1)
    data = yaml.safe_load(ds_file.read_text(encoding="utf-8")) or {}
    return data.get("queries") or []


def build_judge():
    """Cria o juiz LLM."""
    from langchain_openai import ChatOpenAI
    
    def _judge(question, expected, contexts):
        joined = "\n\n---\n\n".join(contexts[:3])
        prompt = (
            "Você é um avaliador especialista em tecnologia de laticínios. "
            "Analise os CONTEXTOS e responda apenas com 'sim' ou 'não' "
            "se eles respondem à QUESTÃO conforme o ESPERADO.\n\n"
            f"QUESTÃO: {question}\nESPERADO: {expected}\n\n"
            f"CONTEXTOS:\n{joined}\n\nSaída: apenas 'sim' ou 'não'."
        )
        llm = ChatOpenAI(
            model=os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
        out = (llm.invoke(prompt).content or "").strip().lower()
        return out.startswith("sim")
    
    return _judge


def check_table_has_data(db_url: str, table_name: str) -> bool:
    """Verifica se a tabela existe e tem dados."""
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                return cur.fetchone()[0] > 0
    except Exception:
        return False


def run_experiment(
    experiment: dict,
    queries: list,
    db_url: str,
    judge,
    max_q: int | None,
) -> list:
    """Executa um experimento para todos os agentes.
    
    Retorna lista de resultados por agente.
    """
    from app.rag.search import search_knowledge_base
    
    name = experiment["name"]
    search_type = experiment.get("search_type", "vector")
    use_hyde = experiment.get("use_hyde", False)
    # reranker seria aplicado após busca — por enquanto testamos retrieval puro
    
    # Agrupa queries por agente
    by_agent = {}
    for q in queries:
        aid = q.get("agent_id", 0)
        if aid not in by_agent:
            by_agent[aid] = []
        by_agent[aid].append(q)
    
    if max_q:
        for aid in by_agent:
            by_agent[aid] = by_agent[aid][:max_q]
    
    results = []
    
    for agent_id in sorted(by_agent.keys()):
        agent_queries = by_agent[agent_id]
        table_name = agent_queries[0].get("table_name", "")
        
        if not check_table_has_data(db_url, table_name):
            continue
        
        hits = 0
        llm_yes = 0
        total = 0
        
        for q in agent_queries:
            pergunta = q.get("pergunta", "")
            expected = q.get("expected")
            expected_all = q.get("expected_all")
            answer = q.get("answer", "")
            
            if not pergunta:
                continue
            total += 1
            
            try:
                res = search_knowledge_base(
                    query=pergunta,
                    table_name=table_name,
                    search_type=search_type,
                    k=5,
                    use_hyde=use_hyde,
                )
            except Exception:
                continue
            
            blob = "\n".join([r.get("content", "") for r in res])
            
            if expected_all:
                ok_hit = all(_norm(t) in _norm(blob) for t in expected_all)
            elif expected:
                ok_hit = _norm(expected) in _norm(blob)
            else:
                ok_hit = len(res) > 0
            hits += 1 if ok_hit else 0
            
            judge_exp = answer or expected or ", ".join(expected_all or [])
            ctxs = [r.get("content", "") for r in res]
            try:
                ok_llm = judge(pergunta, judge_exp, ctxs)
            except Exception:
                ok_llm = False
            llm_yes += 1 if ok_llm else 0
        
        if total > 0:
            results.append({
                "experiment": name,
                "agent_id": agent_id,
                "search_type": search_type,
                "use_hyde": use_hyde,
                "total": total,
                "hits": hits,
                "hit_rate": round(hits / total, 2),
                "llm_yes": llm_yes,
                "llm_rate": round(llm_yes / total, 2),
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RAG Experiments Runner")
    parser.add_argument("--fast", action="store_true", help="Limite de 10 queries")
    parser.add_argument("--max-q", type=int, default=None, help="Limite customizado")
    parser.add_argument("--outfile", type=str, default="experiments", help="Nome base do output")
    args = parser.parse_args()
    
    load_dotenv()
    
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        print("ERRO: SUPABASE_DB_URL não configurada")
        sys.exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERRO: OPENAI_API_KEY não configurada")
        sys.exit(1)
    
    experiments = load_experiments()
    queries = load_dataset()
    judge = build_judge()
    
    max_q = 10 if args.fast else args.max_q
    
    print(f"\nExperimentos: {len(experiments)}")
    print(f"Perguntas: {len(queries)}" + (f" (limitado a {max_q})" if max_q else ""))
    print(f"Início: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    all_results = []
    
    for i, exp in enumerate(experiments, 1):
        name = exp["name"]
        print(f"\n[{i}/{len(experiments)}] {name}...", end=" ", flush=True)
        
        results = run_experiment(exp, queries, db_url, judge, max_q)
        all_results.extend(results)
        
        if results:
            avg_hit = sum(r["hit_rate"] for r in results) / len(results)
            avg_llm = sum(r["llm_rate"] for r in results) / len(results)
            print(f"hit={avg_hit:.0%} llm={avg_llm:.0%}")
        else:
            print("sem dados")
    
    # Salva resultados
    out_dir = Path(__file__).resolve().parents[1] / "tests" / "artifacts" / "rag" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV
    csv_path = out_dir / f"{args.outfile}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    
    # JSON
    json_path = out_dir / f"{args.outfile}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiments_count": len(experiments),
            "queries_count": len(queries),
            "max_q": max_q,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print(f"Resultados salvos em:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    
    # Tabela resumo
    if all_results:
        print(f"\n{'Experimento':>25} | {'Agente':>6} | {'Hit@5':>6} | {'LLM':>6}")
        print("-" * 55)
        for r in all_results:
            print(f"  {r['experiment']:>23} | {r['agent_id']:>6} | {r['hit_rate']:>5.0%} | {r['llm_rate']:>5.0%}")


if __name__ == "__main__":
    main()
