import asyncio
import json

import app.agents.orchestrator as orchestrator_module
from app.agents.orch_text import (
    GENERAL_KNOWLEDGE_TAG,
    OUT_OF_SCOPE_TAG,
    _extract_factual_candidate,
    _is_general_knowledge_response,
    _looks_uncertain,
    _postprocess_consolidated_answer,
)
from app.agents.evidence_reducer import (
    reduce_evidence_for_question,
)
from app.agents.orchestrator import _build_factual_response_candidates
from app.agents.orchestrator import _build_local_primary_regulatory_evidence
from app.agents.orchestrator import _append_missing_regulatory_numeric_complement
from app.agents.orchestrator import _looks_like_chunk_dump
from app.agents.orchestrator import _prefer_direct_fact_response
from app.agents.orchestrator import _requires_synthesis_response
from app.agents.orchestrator import consolidate
from app.agents.orchestrator import _extract_rag_evidence_from_messages
from app.agents.orch_routing import _apply_dairy_hard_constraints
from app.agents.orch_routing import _build_execution_plan
from app.agents.orch_routing import _confidence_to_bucket
from app.agents.orch_routing import _estimate_fastpath_confidence
from app.agents.orch_routing import _rule_based_route
from app.server.webapp import _looks_like_context_dependent_followup
from langchain_core.messages import HumanMessage, ToolMessage


def _build_evidence_grounded_fallback_answer(
    user_text,
    agent_responses,
    general_evidence_text,
    _web_evidence_text,
    _web_sources,
):
    evidence = general_evidence_text or " ".join(
        str(item.get("rag_evidence_text") or item.get("response") or "")
        for item in agent_responses
    )
    reduced = reduce_evidence_for_question(
        user_text,
        evidence,
        agent_id=1,
        top_score=0.08,
        max_sentences=4,
    )
    body = reduced.text or evidence
    return _postprocess_consolidated_answer(
        user_text,
        f"Pontos técnicos principais:\n\n- {body}",
    )


def test_out_of_scope_is_not_factual_candidate():
    assert _extract_factual_candidate(OUT_OF_SCOPE_TAG) is None
    assert _looks_uncertain(OUT_OF_SCOPE_TAG)


def test_general_knowledge_tag_is_valid_factual_content_and_stripped():
    raw = (
        f"{GENERAL_KNOWLEDGE_TAG} O affioramento seleciona parte da microbiota, "
        "reduz gordura e ajuda a equilibrar a sinerese nos queijos Grana."
    )

    assert _is_general_knowledge_response(raw)
    assert _extract_factual_candidate(raw).startswith("O affioramento seleciona")
    assert GENERAL_KNOWLEDGE_TAG not in _postprocess_consolidated_answer("", raw)


def test_deferral_response_is_not_factual_candidate():
    raw = (
        "Nao foram encontrados dados especificos sobre Parmigiano Reggiano. "
        "Recomenda-se consultar estudos ou publicacoes especializadas."
    )

    assert _extract_factual_candidate(raw) is None
    assert _looks_uncertain(raw)


def test_modal_technical_statement_is_not_treated_as_uncertain():
    raw = "A GDL pode acelerar a acidificacao, mas nao substitui o soro-fermento."

    assert not _looks_uncertain(raw)
    assert _extract_factual_candidate(raw) == raw


def test_frozen_and_lyophilized_starter_storage_routes_to_cheese_agent():
    question = "Quais condicoes de armazenamento sao indicadas para fermentos congelados e liofilizados?"

    agent_ids = _rule_based_route(question)
    confidence = _estimate_fastpath_confidence(question, agent_ids or [])
    plan = _build_execution_plan(
        question,
        agent_ids or [],
        [],
        _confidence_to_bucket(confidence),
    )

    assert agent_ids == [1, 3]
    assert plan == [1, 3]


def test_generic_storage_overlap_does_not_make_regulatory_evidence_usable():
    responses = [
        {
            "agent_id": 3,
            "agent_name": "Regulatorios",
            "success": True,
            "response": (
                "O leite pasteurizado pode ser armazenado entre 2 C e 4 C. "
                "Pescado congelado deve ser mantido a -18 C."
            ),
            "rag_evidence_text": (
                "Trecho 1 - score 0.0338 - RIISPOA.md\n"
                "E permitido o armazenamento frigorifico do leite pasteurizado "
                "entre 2 C e 4 C. Durante o transporte, o pescado congelado "
                "deve ser mantido a -18 C."
            ),
            "rag_top_score": 0.0338,
        }
    ]
    question = "Quais condicoes de armazenamento sao indicadas para fermentos congelados e liofilizados?"

    assert _build_factual_response_candidates(responses, question) == []


def test_raw_milk_storage_treatment_uses_specialist_plus_regulatory_route():
    question = (
        "Quando o leite cru ficara armazenado por dois dias ou mais, "
        "qual tratamento e recomendado antes da pasteurizacao convencional?"
    )

    agent_ids = _rule_based_route(question)
    hardened = _apply_dairy_hard_constraints(question, [3])
    plan = _build_execution_plan(question, [3], [], "medium")

    assert agent_ids == [1, 3]
    assert hardened == [1, 3]
    assert plan == [1, 3]


def test_ccs_legislation_routes_to_regulatory_agent():
    assert _rule_based_route("Qual limite de CCS pela legislacao?") == [3]


def test_factual_rag_response_beats_out_of_scope_and_deferral():
    responses = [
        {
            "agent_id": 3,
            "agent_name": "Regulatorios",
            "success": True,
            "response": "Nao ha dados especificos; recomenda-se consultar estudos.",
        },
        {
            "agent_id": 1,
            "agent_name": "Queijos",
            "success": True,
            "response": (
                "A fabricacao tradicional usa tachos de cobre tronco-conicos com "
                "cerca de 1.200 L de leite, favorecendo aquecimento rapido, "
                "decantacao e soldagem dos graos."
            ),
        },
        {
            "agent_id": 0,
            "agent_name": "Base Geral",
            "success": True,
            "response": OUT_OF_SCOPE_TAG,
        },
    ]

    candidates = _build_factual_response_candidates(responses)

    assert len(candidates) == 1
    assert candidates[0]["agent_id"] == 1
    assert "tachos de cobre" in candidates[0]["response"]


def test_rag_evidence_survives_bad_agent_deferral():
    responses = [
        {
            "agent_id": 1,
            "agent_name": "Queijos",
            "success": True,
            "response": "Nao encontrei informacoes suficientes nas fontes internas sobre esse tema.",
            "rag_evidence_text": (
                "Trecho 1 — score 0.0646 — DAIRY_QUEIJOS_DUROS_COMPLETO.md\n"
                "A fabricacao tradicional ocorre em tachos de cobre tronco-conicos "
                "com cerca de 1.200 L de leite."
            ),
        },
        {
            "agent_id": 3,
            "agent_name": "Regulatorios",
            "success": True,
            "response": OUT_OF_SCOPE_TAG,
        },
    ]

    candidates = _build_factual_response_candidates(responses)

    assert len(candidates) == 1
    assert candidates[0]["agent_id"] == 1
    assert candidates[0]["answer_source"] == "rag_evidence"
    assert candidates[0]["requires_consolidation"]
    assert "tachos de cobre" in candidates[0]["response"]


def test_missing_related_regulatory_numeric_limit_is_appended():
    answer = _append_missing_regulatory_numeric_complement(
        "Para Parmesao, a CCS recomendada nao deve superar aproximadamente 300.000 celulas por mL.",
        "Qual contagem de celulas somaticas e recomendada para leite destinado a Parmesao?",
        "Para Parmesao, a CCS recomendada nao deve superar aproximadamente 300.000 celulas por mL.",
        (
            f"{GENERAL_KNOWLEDGE_TAG} Pela legislacao/RIISPOA, o leite cru "
            "refrigerado deve atender ao limite de 500.000 celulas somaticas por mL."
        ),
    )

    assert "300.000" in answer
    assert "500.000" in answer
    assert "regulat" in answer.lower()


def test_primary_specialist_ccs_recommendation_is_locked_over_regulatory_complement():
    question = "Qual contagem de celulas somaticas e recomendada para leite destinado a Parmesao?"
    state = {
        "chosen_agent_ids": [1, 3],
        "messages": [HumanMessage(content=question)],
        "agent_responses": [
            {
                "agent_id": 1,
                "agent_name": "Tecnologia de Queijos",
                "success": True,
                "response": "Nao encontrei informacoes suficientes nas fontes internas sobre esse tema.",
                "rag_evidence_text": (
                    "Trecho 1 - score 0.0846 - DAIRY_QUEIJOS_DUROS_COMPLETO.md\n"
                    "Leite destinado ao Parmesao deve apresentar baixa contagem de celulas somaticas. "
                    "O valor recomendado nao deveria superar aproximadamente 300.000 celulas por mL. "
                    "Contagem elevada esta associada a mastite, menor rendimento e pior aptidao tecnologica."
                ),
                "rag_top_score": 0.0846,
            },
            {
                "agent_id": 3,
                "agent_name": "Regulatorios",
                "success": True,
                "response": OUT_OF_SCOPE_TAG,
            },
        ],
    }

    result = asyncio.run(consolidate(state))
    answer = result["final_response"]

    assert "300.000" in answer
    assert "500.000" in answer
    assert "regulat" in answer.lower()
    assert "nao haja um valor tecnico especifico" not in answer.lower()
    assert "nao ha um valor tecnico especifico" not in answer.lower()


def test_regulatory_primary_ccs_question_can_prioritize_legal_limit():
    question = (
        "Qual contagem de celulas somaticas e recomendada para leite destinado "
        "a Parmesao do ponto de vista da legislacao?"
    )
    state = {
        "chosen_agent_ids": [1, 3],
        "messages": [HumanMessage(content=question)],
        "agent_responses": [
            {
                "agent_id": 1,
                "agent_name": "Tecnologia de Queijos",
                "success": True,
                "response": (
                    "Para Parmesao, a recomendacao tecnologica e nao superar "
                    "aproximadamente 300.000 celulas por mL."
                ),
                "rag_evidence_text": (
                    "Trecho 1 - score 0.0846 - DAIRY_QUEIJOS_DUROS_COMPLETO.md\n"
                    "O valor recomendado nao deveria superar aproximadamente 300.000 celulas por mL."
                ),
                "rag_top_score": 0.0846,
            },
            {
                "agent_id": 3,
                "agent_name": "Regulatorios",
                "success": True,
                "response": (
                    "A Instrucao Normativa MAPA no 76 estabelece que a Contagem de "
                    "Celulas Somaticas deve ser de no maximo 500.000 CS/mL."
                ),
                "rag_evidence_text": (
                    "Trecho 1 - score 0.0700 - IN_76.md\n"
                    "A Contagem de Celulas Somaticas deve ser de no maximo 500.000 CS/mL."
                ),
                "rag_top_score": 0.07,
            },
        ],
    }

    result = asyncio.run(consolidate(state))
    answer = result["final_response"]

    assert "500.000" in answer
    assert (
        "CS/mL" in answer
        or "cs/ml" in answer.lower()
        or "celulas somaticas" in answer.lower()
        or "células somáticas" in answer.lower()
    )


def test_unrelated_generic_riispoa_text_is_not_appended():
    answer = _append_missing_regulatory_numeric_complement(
        "Para Parmesao, a CCS recomendada nao deve superar aproximadamente 300.000 celulas por mL.",
        "Qual contagem de celulas somaticas e recomendada para leite destinado a Parmesao?",
        "Para Parmesao, a CCS recomendada nao deve superar aproximadamente 300.000 celulas por mL.",
        "O RIISPOA classifica infracoes em leves, moderadas, graves e gravissimas no art. 509.",
    )

    assert "509" not in answer


def test_unrelated_regulatory_rag_is_not_usable_evidence_for_specialist_question():
    responses = [
        {
            "agent_id": 3,
            "agent_name": "Regulatorios",
            "success": True,
            "response": "O RIISPOA classifica infracoes em leves, moderadas, graves e gravissimas.",
            "rag_evidence_text": (
                "Trecho 1 — score 0.0700 — RIISPOA.md\n"
                "O RIISPOA classifica infracoes em leves, moderadas, graves e gravissimas no art. 509."
            ),
            "rag_top_score": 0.07,
        }
    ]

    candidates = _build_factual_response_candidates(
        responses,
        (
            "Qual contagem de celulas somaticas e recomendada para leite destinado "
            "a Parmesao? E do ponto de vista da legislacao?"
        ),
    )

    assert candidates == []


def test_related_regulatory_rag_can_complement_specialist_question_without_legal_wording():
    responses = [
        {
            "agent_id": 3,
            "agent_name": "Regulatorios",
            "success": True,
            "response": "",
            "rag_evidence_text": (
                "Trecho 1 - score 0.0710 - RTIQ_Parmesao.md\n"
                "O queijo Parmesao e classificado como queijo duro de baixa umidade, "
                "com umidade maxima de 35,9% quando completamente curado."
            ),
            "rag_top_score": 0.071,
        }
    ]

    candidates = _build_factual_response_candidates(
        responses,
        "Qual rendimento deve ser considerado para Parmesao logo apos a salga e apos cerca de 12 meses?",
    )

    assert len(candidates) == 1
    assert candidates[0]["agent_id"] == 3
    assert "Parmesao" in candidates[0]["response"]


def test_maturation_rule_does_not_answer_ccs_legal_followup():
    responses = [
        {
            "agent_id": 3,
            "agent_name": "Regulatorios",
            "success": True,
            "response": (
                "A legislacao permite leite cru para queijos maturados por no "
                "minimo 60 dias."
            ),
            "rag_evidence_text": (
                "RIISPOA Art. 373. Queijos elaborados a partir de leite cru devem "
                "ser maturados por periodo nao inferior a sessenta dias."
            ),
            "rag_top_score": 0.09,
        }
    ]

    candidates = _build_factual_response_candidates(
        responses,
        (
            "Qual contagem de celulas somaticas e recomendada para leite destinado "
            "a Parmesao? E do ponto de vista da legislacao?"
        ),
    )

    assert candidates == []


def test_local_primary_regulatory_ccs_source_is_available():
    evidence = _build_local_primary_regulatory_evidence(
        "e do ponto de vista da legislacao? Qual contagem de celulas somaticas?"
    )

    assert evidence is not None
    assert "Instrucao Normativa MAPA no 76" in evidence
    assert "500.000" in evidence
    assert "Celulas Somaticas" in evidence


def test_relevant_regulatory_rag_is_usable_without_specialist_evidence():
    responses = [
        {
            "agent_id": 3,
            "agent_name": "Regulatorios",
            "success": True,
            "response": (
                "O leite cru refrigerado deve atender ao limite de "
                "500.000 celulas somaticas por mL."
            ),
            "rag_evidence_text": (
                "Trecho 1 — score 0.0700 — IN_76.md\n"
                "O leite cru refrigerado deve atender ao limite de "
                "500.000 celulas somaticas por mL."
            ),
            "rag_top_score": 0.07,
        }
    ]
    question = (
        "Qual contagem de celulas somaticas e recomendada para leite destinado "
        "a Parmesao do ponto de vista da legislacao?"
    )

    candidates = _build_factual_response_candidates(responses, question)

    assert len(candidates) == 1
    assert candidates[0]["evidence_quality"] == "usable_regulatory_evidence"


def test_short_e_do_followup_uses_conversation_context():
    assert _looks_like_context_dependent_followup("e do ponto de vista da legislacao?")


def test_tool_messages_are_promoted_to_rag_evidence():
    msg = ToolMessage(
        content=(
            '[{"content":"Tanques tronco-conicos de cobre favorecem decantacao '
            'e soldagem dos graos.","score":0.0646,'
            '"metadata":{"source":"DAIRY_QUEIJOS_DUROS_COMPLETO.md"}}]'
        ),
        tool_call_id="search_1",
    )

    evidence_text, rows = _extract_rag_evidence_from_messages([msg])

    assert len(rows) == 1
    assert "Tanques tronco-conicos de cobre" in evidence_text
    assert "DAIRY_QUEIJOS_DUROS_COMPLETO.md" in evidence_text


def test_tool_message_wrapped_results_are_promoted_to_rag_evidence():
    msg = ToolMessage(
        content=json.dumps({
            "results": [
                {
                    "text": (
                        "A temperatura recomendada de cura situa-se entre 16 e 18 C. "
                        "Quando o queijo e maturado abaixo de 16 C, as reacoes bioquimicas ficam lentas."
                    ),
                    "relevance_score": 0.1042,
                    "source": "DAIRY_QUEIJOS_DUROS_COMPLETO.md",
                }
            ]
        }),
        tool_call_id="search_1",
    )

    evidence_text, rows = _extract_rag_evidence_from_messages([msg])

    assert len(rows) == 1
    assert rows[0]["score"] == 0.1042
    assert "16 e 18 C" in evidence_text
    assert "DAIRY_QUEIJOS_DUROS_COMPLETO.md" in evidence_text


def test_wrapped_rag_evidence_prevents_false_zero_evidence_answer():
    question = "Qual faixa de temperatura e recomendada para maturar Parmesao com desenvolvimento adequado de grana, sabor e aroma?"
    msg = ToolMessage(
        content={
            "results": [
                {
                    "page_content": (
                        "A temperatura recomendada de cura situa-se entre 16 e 18 C. "
                        "Temperaturas baixas reduzem o metabolismo bacteriano, a proteolise e a lipolise."
                    ),
                    "score": 0.1042,
                    "metadata": {"source": "DAIRY_QUEIJOS_DUROS_COMPLETO.md"},
                }
            ]
        },
        tool_call_id="search_1",
    )
    evidence_text, rows = _extract_rag_evidence_from_messages([msg])
    candidates = _build_factual_response_candidates(
        [
            {
                "agent_id": 1,
                "agent_name": "Tecnologia de Queijos",
                "success": True,
                "response": "[FORA_DE_ESCOPO]",
                "rag_evidence_text": evidence_text,
                "rag_top_score": rows[0]["score"],
            }
        ],
        question,
    )

    result = asyncio.run(consolidate({
        "chosen_agent_ids": [1, 3],
        "messages": [HumanMessage(content=question)],
        "agent_responses": candidates,
    }))

    answer_lower = result["final_response"].lower()
    assert "16" in answer_lower and "18" in answer_lower, f"Expected 16-18°C range in: {result['final_response'][:200]}"
    assert "Nao ha informacoes disponiveis" not in result["final_response"]


def test_consolidation_failure_fallback_returns_clean_extractive_answer():
    answer = _build_evidence_grounded_fallback_answer(
        "Quais fatores tornam a mussarela mais propensa a gosto amargo?",
        [
            {
                "agent_id": 1,
                "agent_name": "Queijos",
                "success": True,
                "response": "",
                "rag_evidence_text": (
                    "Trecho 1 — score 0.0731 — DAIRY_MUSSARELA_COMPLETO.md\n"
                    "**35. Conjunto de fatores favoráveis ao amargor** "
                    "|Fator|Condição crítica| |---|---| "
                    "|Pasteurização alta|75 °C ou mais| "
                    "|Excesso de coagulante|Maior formação de peptídeos| "
                    "O sabor amargo ocorre por acúmulo de peptídeos de baixo peso molecular. "
                    "Baixo sal reduz o controle da proteólise e baixo teor de gordura reduz o mascaramento sensorial. "
                    "Leite com alta contagem de psicrotróficos gera proteases termorresistentes."
                ),
            }
        ],
        "",
        "",
        [],
    )

    assert "consolidação automática falhou" not in answer.lower()
    assert "Trecho 1" not in answer
    assert "|---|" not in answer
    assert "Pontos técnicos principais" in answer
    assert "peptídeos" in answer or "peptideos" in answer
    assert "75 °C" in answer


def test_internal_rag_metadata_never_reaches_final_answer():
    raw = (
        "Pontos técnicos principais:\n\n"
        "- [embeddings_agente_1_queijos] casca Quando a Mussarela entra quente "
        "na salmoura, absorve sal rapidamente na periferia.\n"
        "- source_table=embeddings_agente_1_queijos A diferença de umidade pode "
        "chegar a 4%."
    )

    answer = _postprocess_consolidated_answer(
        "Como a entrada quente da mussarela na salmoura pode provocar casca mole?",
        raw,
    )

    assert "embeddings_agente" not in answer
    assert "source_table" not in answer
    assert "Mussarela entra quente" in answer
    assert "4%" in answer


def test_evidence_reducer_extracts_parmesan_yield_without_context_bleed():
    question = "Qual rendimento deve ser considerado para Parmesao logo apos a salga e apos cerca de 12 meses?"
    evidence = (
        "Trecho 1 - score 0.0842 - DAIRY_QUEIJOS_DUROS_COMPLETO.md\n"
        "**8. Expectativa de rendimento** O Parmesao apresenta rendimento inferior ao de queijos moles e semiduros. "
        "Isso ocorre por varios fatores. Logo apos a salga, o rendimento pode ficar em torno de "
        "12 litros de leite por kg de queijo. Quando o queijo e maturado por pelo menos 12 meses, "
        "o rendimento pode chegar a aproximadamente 15 litros de leite por kg de queijo. "
        "Regulamento citado: Provolone, Minas Padrao e Ricota."
    )

    reduced = reduce_evidence_for_question(question, evidence, agent_id=1, top_score=0.0842)

    assert reduced.direct_answer is not None
    assert "12 litros" in reduced.direct_answer
    assert "15 litros" in reduced.direct_answer
    assert "Provolone" not in reduced.direct_answer
    assert "Minas" not in reduced.direct_answer


def test_evidence_reducer_extracts_parmesan_maturation_temperature():
    question = "Qual faixa de temperatura e recomendada para maturar Parmesao com desenvolvimento adequado de grana, sabor e aroma?"
    evidence = (
        "Trecho 1 - score 0.1042 - DAIRY_QUEIJOS_DUROS_COMPLETO.md\n"
        "**4. Maturacao como eixo central da identidade do Parmesao** "
        "A maturacao ideal para um Parmesao de melhor qualidade deve ser de aproximadamente 12 meses. "
        "A temperatura recomendada de cura situa-se entre 16 e 18 C. "
        "Quando o queijo e maturado abaixo de 16 C, as reacoes bioquimicas ficam muito lentas. "
        "Nessas condicoes, 6 meses nao sao suficientes para formacao de textura granulosa nem para "
        "desenvolvimento adequado de sabor e aroma."
    )

    reduced = reduce_evidence_for_question(question, evidence, agent_id=1, top_score=0.1042)

    assert reduced.direct_answer == "A temperatura recomendada de cura situa-se entre 16 e 18 C."


def test_evidence_reducer_rejects_numeric_sentence_missing_explicit_anchor():
    question = "Quando o vacuo for inevitavel em Parmesao, qual condicao de umidade deve ser respeitada?"
    evidence = (
        "Trecho 1 - score 0.0446 - DAIRY_QUEIJOS_DUROS_COMPLETO.md\n"
        "Parmesao ralado: particulas de 1-2 mm; umidade antes da secagem de 32-35%; "
        "ar quente a 50-55 C; secagem por 20-25 minutos; umidade final de 14-19%; "
        "atmosfera modificada com O2 maximo de 1,5%."
    )

    reduced = reduce_evidence_for_question(question, evidence, agent_id=1, top_score=0.0446)

    assert reduced.text == ""
    assert reduced.direct_answer is None


def test_consolidate_uses_rag_evidence_even_when_agent_answer_defers():
    question = "Qual faixa de temperatura e recomendada para maturar Parmesao com desenvolvimento adequado de grana, sabor e aroma?"
    state = {
        "chosen_agent_ids": [1],
        "messages": [HumanMessage(content=question)],
        "agent_responses": [
            {
                "agent_id": 1,
                "agent_name": "Tecnologia de Queijos",
                "success": True,
                "response": "Nao encontrei informacoes suficientes nas fontes internas sobre esse tema.",
                "rag_evidence_text": (
                    "Trecho 1 - score 0.1042 - DAIRY_QUEIJOS_DUROS_COMPLETO.md\n"
                    "A temperatura recomendada de cura situa-se entre 16 e 18 C. "
                    "Quando o queijo e maturado abaixo de 16 C, as reacoes bioquimicas ficam muito lentas."
                ),
                "rag_top_score": 0.1042,
            }
        ],
    }

    result = asyncio.run(consolidate(state))

    answer_lower = result["final_response"].lower()
    assert "16" in answer_lower and "18" in answer_lower, f"Expected 16-18°C range in: {result['final_response'][:200]}"
    assert "Nao ha informacoes disponiveis" not in result["final_response"]


def test_consolidate_answers_short_numeric_fact_deterministically(monkeypatch):
    async def fail_if_llm_called(_state, _prompt):
        raise AssertionError("short numeric facts should not depend on consolidation LLM")

    async def fail_if_web_called(_user_text):
        raise AssertionError("web fallback should not run when KB has direct numeric evidence")

    monkeypatch.setattr(orchestrator_module, "_ainvoke_consolidation_with_timeout", fail_if_llm_called)
    monkeypatch.setattr(orchestrator_module, "_fetch_web_fallback_evidence", fail_if_web_called)

    question = (
        "Qual faixa de temperatura e recomendada para maturar Parmesao "
        "com desenvolvimento adequado de grana, sabor e aroma?"
    )
    state = {
        "chosen_agent_ids": [1, 3],
        "messages": [HumanMessage(content=question)],
        "agent_responses": [
            {
                "agent_id": 1,
                "agent_name": "Tecnologia de Queijos",
                "success": True,
                "response": "Nao encontrei informacoes suficientes nas fontes internas sobre esse tema.",
                "rag_evidence_text": (
                    "Trecho 1 - score 0.1042 - DAIRY_QUEIJOS_DUROS_COMPLETO.md\n"
                    "A maturacao ideal para um Parmesao de melhor qualidade deve ser de aproximadamente 12 meses. "
                    "A temperatura recomendada de cura situa-se entre 16 e 18 °C. "
                    "Quando o queijo e maturado abaixo de 16 °C, as reacoes bioquimicas ficam muito lentas. "
                    "Nessas condicoes, 6 meses nao sao suficientes para formacao de textura granulosa nem para "
                    "desenvolvimento adequado de sabor e aroma."
                ),
                "rag_top_score": 0.1042,
            },
            {
                "agent_id": 3,
                "agent_name": "Regulatorios",
                "success": True,
                "response": OUT_OF_SCOPE_TAG,
            },
        ],
    }

    result = asyncio.run(consolidate(state))

    answer = result["final_response"]
    assert "16" in answer and "18" in answer
    assert "°C" in answer or "C" in answer
    assert "Nao encontrei" not in answer
    assert "nao encontrei" not in answer.lower()


def test_web_last_resort_runs_only_when_kb_has_no_usable_evidence(monkeypatch):
    async def fake_fetch(user_text):
        assert "queijo" in user_text.lower()
        return (
            "[Fonte 1] Artigo tecnico (example.org) - Evidencia externa sobre o tema consultado.",
            [{"title": "Artigo tecnico", "domain": "example.org", "url": "https://example.org/tema"}],
            "web_fallback_evidence_collected",
        )

    async def fake_llm(_state, prompt):
        assert "EVIDENCIAS WEB" in prompt
        return "A evidencia externa consultada indica um ponto tecnico sustentado sobre o tema."

    monkeypatch.setattr(orchestrator_module, "_fetch_web_fallback_evidence", fake_fetch)
    monkeypatch.setattr(orchestrator_module, "_ainvoke_consolidation_with_timeout", fake_llm)

    state = {
        "chosen_agent_ids": [1, 3],
        "messages": [HumanMessage(content="O que dizem fontes tecnicas sobre queijo colonial maturado?")],
        "agent_responses": [
            {
                "agent_id": 1,
                "agent_name": "Tecnologia de Queijos",
                "success": True,
                "response": OUT_OF_SCOPE_TAG,
            },
            {
                "agent_id": 3,
                "agent_name": "Regulatorios",
                "success": True,
                "response": OUT_OF_SCOPE_TAG,
            },
        ],
    }

    result = asyncio.run(consolidate(state))

    assert result["web_fallback_used"] is True
    assert result["fallback_used"] is True
    assert "evidencia externa consultada" in result["final_response"].lower()
    assert "Fonte consultada:" in result["final_response"]
    assert "https://example.org/tema" in result["final_response"]


def test_consolidate_does_not_promote_raw_chunks_when_evidence_filter_rejects(monkeypatch):
    async def fake_fetch(_user_text):
        return "", [], "web_empty"

    async def fail_if_called(_state, prompt):
        raise AssertionError(f"raw chunk should not reach synthesis prompt: {prompt[:200]}")

    monkeypatch.setattr(orchestrator_module, "_fetch_web_fallback_evidence", fake_fetch)
    monkeypatch.setattr(orchestrator_module, "_ainvoke_consolidation_with_timeout", fail_if_called)

    state = {
        "chosen_agent_ids": [1, 3],
        "messages": [HumanMessage(content="Qual risco existe em usar cultura de iogurte para Parmesao?")],
        "agent_responses": [
            {
                "agent_id": 1,
                "agent_name": "Tecnologia de Queijos",
                "success": True,
                "response": "",
                "rag_evidence_text": (
                    "Trecho 1 - score 0.0100 - RIISPOA.md\n"
                    "O pescado congelado deve ser mantido a -18 C durante transporte."
                ),
                "rag_top_score": 0.01,
            }
        ],
    }

    result = asyncio.run(consolidate(state))

    assert "Trecho 1" not in result["final_response"]
    assert "pescado" not in result["final_response"].lower()
    assert result["web_fallback_used"] is False


def test_chunk_dump_detector_flags_internal_rag_artifacts():
    assert _looks_like_chunk_dump("Trecho 1 - score 0.0642 - fonte.md\nconteudo")
    assert _looks_like_chunk_dump("source_table=embeddings_agente_1_queijos resposta")


def test_web_last_resort_does_not_run_when_specialist_has_kb_evidence(monkeypatch):
    async def fail_if_called(_user_text):
        raise AssertionError("web fallback should not run when KB evidence exists")

    monkeypatch.setattr(orchestrator_module, "_fetch_web_fallback_evidence", fail_if_called)

    question = "Qual contagem de celulas somaticas e recomendada para leite destinado a Parmesao?"
    state = {
        "chosen_agent_ids": [1, 3],
        "messages": [HumanMessage(content=question)],
        "agent_responses": [
            {
                "agent_id": 1,
                "agent_name": "Tecnologia de Queijos",
                "success": True,
                "response": OUT_OF_SCOPE_TAG,
                "rag_evidence_text": (
                    "Trecho 1 - score 0.0846 - DAIRY_QUEIJOS_DUROS_COMPLETO.md\n"
                    "O valor recomendado nao deveria superar aproximadamente 300.000 celulas por mL."
                ),
                "rag_top_score": 0.0846,
            },
            {
                "agent_id": 3,
                "agent_name": "Regulatorios",
                "success": True,
                "response": OUT_OF_SCOPE_TAG,
            },
        ],
    }

    result = asyncio.run(consolidate(state))

    assert "300.000" in result["final_response"]
    assert not result.get("web_fallback_used", False)


def test_technical_cheese_questions_route_regulatory_as_complement():
    for question in (
        "Qual rendimento deve ser considerado para Parmesao logo apos a salga e apos cerca de 12 meses?",
        "Qual faixa de temperatura e recomendada para maturar Parmesao com desenvolvimento adequado de grana, sabor e aroma?",
    ):
        ids = _rule_based_route(question)
        confidence = _estimate_fastpath_confidence(question, ids or [])
        plan = _build_execution_plan(question, ids or [], [], _confidence_to_bucket(confidence))

        assert ids == [1, 3]
        assert plan == [1, 3]


def test_composition_and_process_question_requires_synthesis_not_direct_fragment():
    question = "Qual composicao e processo basico sao indicados para queijo do Reino curado?"
    responses = [
        {
            "agent_id": 1,
            "agent_name": "Tecnologia de Queijos",
            "success": True,
            "response": (
                "helveticus_ , coagulacao 32 C a 35 C por 25 min a 40 min, "
                "graos pequenos, cozimento 43 C a 45 C, maturacao tradicional."
            ),
        }
    ]

    assert _requires_synthesis_response(question)
    assert _prefer_direct_fact_response(question, responses) is None


def test_postprocess_removes_stray_markdown_underscores():
    answer = _postprocess_consolidated_answer(
        "Qual composicao e processo basico sao indicados para queijo do Reino curado?",
        "Lactobacillus helveticus_ e indicado no processo.",
    )

    assert "helveticus_" not in answer
    assert "helveticus" in answer


def test_extractive_fallback_strips_general_index_source_table_prefix():
    answer = _build_evidence_grounded_fallback_answer(
        "Como a entrada quente da mussarela na salmoura pode provocar casca mole?",
        [],
        (
            "[embeddings_agente_1_queijos] casca Quando a Mussarela entra quente "
            "na salmoura, sem banho prévio de água gelada ou fria, absorve sal "
            "rapidamente na periferia. A diferença de umidade entre centro e "
            "periferia pode chegar a 4%, causando amolecimento perceptível da casca."
        ),
        "",
        [],
    )

    assert "embeddings_agente" not in answer
    assert "Pontos técnicos principais" in answer
    assert "absorve sal rapidamente" in answer
    assert "4%" in answer
