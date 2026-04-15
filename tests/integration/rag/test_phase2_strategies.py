"""
tests/integration/rag/test_phase2_strategies.py — Fase 2: Comparação de estratégias

Compara as 3 estratégias de busca (vector, text, hybrid_rrf) para
cada agente, usando o mesmo dataset de perguntas.

Objetivo: identificar qual estratégia funciona melhor para cada agente.
Hipótese: hybrid_rrf é melhor para agentes 3 e 5 (regulatórios e defeitos)
porque esses domínios têm termos exatos (IN 76, Clostridium).

Equivalente ao test_retrieval_strategies.py e test_retrieval_eval_llm.py
do original, combinados e adaptados.

Uso:
  make rag_phase2_fast  (10 perguntas por agente)
  make rag_phase2       (todas as perguntas)
"""

import os
import unicodedata
import pytest
import psycopg

pytestmark = pytest.mark.phase2


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


SEARCH_TYPES = ["vector", "text", "hybrid_rrf"]


def test_phase2_compare_strategies(
    db_supabase, require_openai, rag_dataset, max_queries, llm_judge_yesno
):
    """Compara vector vs text vs hybrid_rrf para cada agente."""
    from app.rag.search import search_knowledge_base
    
    # Agrupa por agente
    queries_by_agent = {}
    for q in rag_dataset:
        aid = q.get("agent_id", 0)
        if aid not in queries_by_agent:
            queries_by_agent[aid] = []
        queries_by_agent[aid].append(q)
    
    if max_queries:
        for aid in queries_by_agent:
            queries_by_agent[aid] = queries_by_agent[aid][:max_queries]
    
    print("\n" + "=" * 70)
    print("FASE 2 — COMPARAÇÃO DE ESTRATÉGIAS DE BUSCA")
    print("=" * 70)
    
    all_results = []
    
    for agent_id in sorted(queries_by_agent.keys()):
        queries = queries_by_agent[agent_id]
        table_name = queries[0].get("table_name", "")
        
        # Verifica se a tabela tem dados
        try:
            with psycopg.connect(db_supabase) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    if cur.fetchone()[0] == 0:
                        print(f"\n  Agente {agent_id}: tabela vazia — pulando")
                        continue
        except Exception:
            print(f"\n  Agente {agent_id}: tabela não existe — pulando")
            continue
        
        agent_results = {}
        
        for search_type in SEARCH_TYPES:
            hits = 0
            llm_yes = 0
            total = 0
            
            for q in queries:
                pergunta = q.get("pergunta", "")
                expected = q.get("expected")
                expected_all = q.get("expected_all")
                answer = q.get("answer", "")
                
                if not pergunta:
                    continue
                total += 1
                
                try:
                    results = search_knowledge_base(
                        query=pergunta,
                        table_name=table_name,
                        search_type=search_type,
                        k=5,
                    )
                except Exception:
                    # text search pode falhar se FTS não está configurado
                    continue
                
                blob = "\n".join([r.get("content", "") for r in results])
                
                # Hit@k
                if expected_all:
                    ok_hit = all(_norm(t) in _norm(blob) for t in expected_all)
                elif expected:
                    ok_hit = _norm(expected) in _norm(blob)
                else:
                    ok_hit = len(results) > 0
                hits += 1 if ok_hit else 0
                
                # LLM judge
                judge_exp = answer or expected or ", ".join(expected_all or [])
                ctxs = [r.get("content", "") for r in results]
                try:
                    ok_llm = llm_judge_yesno(pergunta, judge_exp, ctxs)
                except Exception:
                    ok_llm = False
                llm_yes += 1 if ok_llm else 0
            
            hit_rate = round(hits / max(total, 1), 2)
            llm_rate = round(llm_yes / max(total, 1), 2)
            
            agent_results[search_type] = {
                "hit_rate": hit_rate,
                "llm_rate": llm_rate,
                "total": total,
            }
        
        all_results.append({
            "agent_id": agent_id,
            "table": table_name,
            "results": agent_results,
        })
        
        # Imprime resultados do agente
        print(f"\n  Agente {agent_id} ({table_name})")
        print(f"    {'Estratégia':>12} | {'Hit@5':>6} | {'LLM':>6}")
        print(f"    {'-' * 32}")
        for st in SEARCH_TYPES:
            r = agent_results.get(st, {})
            print(f"    {st:>12} | {r.get('hit_rate', 0):>5.0%} | {r.get('llm_rate', 0):>5.0%}")
    
    # Resumo comparativo
    if all_results:
        print("\n" + "=" * 70)
        print("RESUMO COMPARATIVO (Hit@5)")
        print(f"  {'Agente':>8} | {'vector':>8} | {'text':>8} | {'hybrid':>8} | Melhor")
        print(f"  {'-' * 55}")
        
        for ar in all_results:
            res = ar["results"]
            v = res.get("vector", {}).get("hit_rate", 0)
            t = res.get("text", {}).get("hit_rate", 0)
            h = res.get("hybrid_rrf", {}).get("hit_rate", 0)
            
            best = max([(v, "vector"), (t, "text"), (h, "hybrid")], key=lambda x: x[0])
            
            print(f"  {ar['agent_id']:>8} | {v:>7.0%} | {t:>7.0%} | {h:>7.0%} | {best[1]}")
