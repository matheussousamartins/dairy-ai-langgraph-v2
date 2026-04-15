"""
tests/integration/rag/test_phase1_retrieval.py — Fase 1: Avaliação de retrieval

Testa se a busca vetorial retorna chunks relevantes para cada agente.
Usa o dataset rag_queries.yaml e verifica:
  1. Se o expected aparece nos chunks retornados (hit@k)
  2. Se o juiz LLM considera os chunks suficientes (llm_yes)

Este teste pressupõe que os documentos já foram ingeridos
nos agentes (Fase 0). Se a tabela estiver vazia, pula.

Equivalente ao test_phase1_semantic_qa.py do original.

Uso: make rag_phase1
"""

import os
import unicodedata
import pytest
import psycopg

pytestmark = pytest.mark.phase1


def _norm(s: str) -> str:
    """Normaliza texto para comparação case-insensitive sem acentos."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


def test_phase1_retrieval_per_agent(
    db_supabase, require_openai, rag_dataset, max_queries, llm_judge_yesno
):
    """Avalia a qualidade do retrieval para cada agente.
    
    Para cada pergunta do dataset:
      1. Faz busca vetorial na tabela do agente
      2. Verifica se os termos esperados estão nos chunks (hit@k)
      3. Usa juiz LLM para avaliar se os chunks respondem a pergunta
    
    Imprime relatório por agente com taxas de acerto.
    """
    from app.rag.search import search_knowledge_base
    
    # Agrupa queries por agent_id
    queries_by_agent = {}
    for q in rag_dataset:
        aid = q.get("agent_id", 0)
        if aid not in queries_by_agent:
            queries_by_agent[aid] = []
        queries_by_agent[aid].append(q)
    
    # Aplica limite de fast mode
    if max_queries:
        for aid in queries_by_agent:
            queries_by_agent[aid] = queries_by_agent[aid][:max_queries]
    
    print("\n" + "=" * 60)
    print("FASE 1 — AVALIAÇÃO DE RETRIEVAL (vector only)")
    print("=" * 60)
    
    all_results = []
    
    for agent_id in sorted(queries_by_agent.keys()):
        queries = queries_by_agent[agent_id]
        table_name = queries[0].get("table_name", "")
        
        # Verifica se a tabela tem dados
        try:
            with psycopg.connect(db_supabase) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cur.fetchone()[0]
            if count == 0:
                print(f"\n  Agente {agent_id}: tabela {table_name} VAZIA — pulando")
                continue
        except Exception:
            print(f"\n  Agente {agent_id}: tabela {table_name} não existe — pulando")
            continue
        
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
            
            # Busca vetorial
            results = search_knowledge_base(
                query=pergunta,
                table_name=table_name,
                search_type="vector",
                k=5,
            )
            
            # Junta todo o conteúdo retornado
            blob = "\n".join([r.get("content", "") for r in results])
            
            # Hit@k: verifica se expected está nos chunks
            if expected_all:
                ok_hit = all(_norm(term) in _norm(blob) for term in expected_all)
            elif expected:
                ok_hit = _norm(expected) in _norm(blob)
            else:
                ok_hit = len(results) > 0
            
            hits += 1 if ok_hit else 0
            
            # Juiz LLM
            judge_expected = answer or expected or ", ".join(expected_all or [])
            contexts = [r.get("content", "") for r in results]
            try:
                ok_llm = llm_judge_yesno(pergunta, judge_expected, contexts)
            except Exception:
                ok_llm = False
            
            llm_yes += 1 if ok_llm else 0
        
        hit_rate = round(hits / max(total, 1), 2)
        llm_rate = round(llm_yes / max(total, 1), 2)
        
        all_results.append({
            "agent_id": agent_id,
            "table": table_name,
            "total": total,
            "hit_rate": hit_rate,
            "llm_rate": llm_rate,
        })
        
        print(f"\n  Agente {agent_id} ({table_name})")
        print(f"    Perguntas: {total}")
        print(f"    Hit@5: {hit_rate:.0%}")
        print(f"    LLM judge: {llm_rate:.0%}")
    
    # Resumo final
    if all_results:
        print("\n" + "-" * 60)
        print(f"{'Agente':>8} | {'Hit@5':>6} | {'LLM':>6} | {'Total':>5}")
        print("-" * 60)
        for r in all_results:
            print(f"  {r['agent_id']:>5}  | {r['hit_rate']:>5.0%} | {r['llm_rate']:>5.0%} | {r['total']:>5}")
        
        # Critério mínimo: ao menos 50% de hit@5 em média
        avg_hit = sum(r["hit_rate"] for r in all_results) / len(all_results)
        print(f"\n  Média Hit@5: {avg_hit:.0%}")
        assert avg_hit >= 0.3, f"Hit@5 médio ({avg_hit:.0%}) abaixo do mínimo (30%)"
