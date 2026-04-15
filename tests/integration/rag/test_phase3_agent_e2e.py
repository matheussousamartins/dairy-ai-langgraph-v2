"""
tests/integration/rag/test_phase3_agent_e2e.py — Fase 3: Avaliação end-to-end do agente

Testa o agente COMPLETO (não apenas o retrieval): pergunta → agente →
resposta. O juiz LLM avalia se a resposta final do agente é adequada.

Diferença da Fase 1 e 2:
  - Fase 1: avalia se os CHUNKS retornados são relevantes
  - Fase 2: compara ESTRATÉGIAS de busca
  - Fase 3: avalia a RESPOSTA FINAL do agente (inclui prompt, RAG, LLM)

Se o retrieval estiver perfeito mas o prompt for ruim, a Fase 3 pega.
Se o prompt for perfeito mas o retrieval estiver ruim, a Fase 1 pega.

Equivalente ao test_phase3_agent_dataset.py do original.

Uso:
  make rag_phase3_fast  (10 perguntas)
  make rag_phase3       (todas)
"""

import os
import pytest
import psycopg
from langchain_core.messages import HumanMessage

pytestmark = pytest.mark.phase3


def test_phase3_agent_e2e(
    db_supabase, require_openai, rag_dataset, max_queries, llm_judge_yesno
):
    """Executa cada agente sobre o dataset e avalia com juiz LLM.
    
    Para cada pergunta:
      1. Invoca o grafo do agente correspondente
      2. Extrai a resposta final (última AIMessage)
      3. Juiz LLM avalia se a resposta contém o esperado
    
    Imprime relatório por agente e geral.
    """
    from app.agents.base_agent import get_agent_graph
    
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
    
    print("\n" + "=" * 60)
    print("FASE 3 — AVALIAÇÃO END-TO-END DO AGENTE")
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
                    if cur.fetchone()[0] == 0:
                        print(f"\n  Agente {agent_id}: tabela vazia — pulando")
                        continue
        except Exception:
            print(f"\n  Agente {agent_id}: tabela não existe — pulando")
            continue
        
        # Carrega o grafo do agente
        try:
            graph = get_agent_graph(agent_id)
        except Exception as e:
            print(f"\n  Agente {agent_id}: erro ao carregar grafo — {e}")
            continue
        
        total = 0
        llm_yes = 0
        errors = 0
        
        for q in queries:
            pergunta = q.get("pergunta", "")
            expected = q.get("expected")
            expected_all = q.get("expected_all")
            answer = q.get("answer", "")
            
            if not pergunta:
                continue
            total += 1
            
            # Invoca o agente
            try:
                result = graph.invoke({
                    "messages": [HumanMessage(content=pergunta)]
                })
                
                # Extrai resposta final
                messages = result.get("messages", [])
                agent_answer = ""
                for msg in reversed(messages):
                    if hasattr(msg, "content"):
                        content = msg.content
                        if isinstance(content, list):
                            parts = [
                                p.get("text", "")
                                for p in content
                                if isinstance(p, dict)
                            ]
                            agent_answer = "\n".join(parts)
                        elif isinstance(content, str):
                            agent_answer = content
                        if agent_answer:
                            break
                
            except Exception as e:
                print(f"    ERRO: {pergunta[:50]}... → {e}")
                errors += 1
                continue
            
            # Juiz LLM avalia a resposta
            judge_expected = answer or expected or ", ".join(expected_all or [])
            try:
                ok = llm_judge_yesno(pergunta, judge_expected, [agent_answer])
            except Exception:
                ok = False
            
            llm_yes += 1 if ok else 0
        
        rate = round(llm_yes / max(total, 1), 2)
        
        all_results.append({
            "agent_id": agent_id,
            "total": total,
            "llm_yes": llm_yes,
            "rate": rate,
            "errors": errors,
        })
        
        print(f"\n  Agente {agent_id}")
        print(f"    Perguntas: {total}")
        print(f"    LLM approved: {rate:.0%} ({llm_yes}/{total})")
        if errors:
            print(f"    Erros: {errors}")
    
    # Resumo
    if all_results:
        print("\n" + "=" * 60)
        print("RESUMO FASE 3")
        print(f"  {'Agente':>8} | {'Aprovado':>10} | {'Total':>5} | {'Erros':>5}")
        print(f"  {'-' * 40}")
        for r in all_results:
            print(f"  {r['agent_id']:>8} | {r['rate']:>9.0%} | {r['total']:>5} | {r['errors']:>5}")
        
        avg_rate = sum(r["rate"] for r in all_results) / len(all_results)
        print(f"\n  Média aprovação: {avg_rate:.0%}")
        
        assert avg_rate >= 0.5, (
            f"Taxa média de aprovação ({avg_rate:.0%}) abaixo do mínimo (50%)"
        )
