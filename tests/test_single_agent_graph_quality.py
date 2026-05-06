from app.graphs.single_agent_graph import _reduce_chunks_for_prompt
from app.agents.orch_quality import QuestionType, detect_question_type
from app.agents.synthesis_rules import FORMAT_INSTRUCTIONS
from app.rag.metadata_filters import classify_query_intent


def test_single_agent_reduces_evidence_before_synthesis_prompt():
    question = "Qual faixa de temperatura e recomendada para maturar Parmesao com desenvolvimento adequado de grana, sabor e aroma?"
    chunks = [
        {
            "content": (
                "Regulamento citado: Provolone, Minas Padrao e Ricota. "
                "A maturacao ideal para um Parmesao de melhor qualidade deve ser de aproximadamente 12 meses. "
                "A temperatura recomendada de cura situa-se entre 16 e 18 C. "
                "Quando o queijo e maturado abaixo de 16 C, as reacoes bioquimicas ficam muito lentas."
            ),
            "score": 0.1042,
        }
    ]

    reduced, stats = _reduce_chunks_for_prompt(
        question=question,
        chunks=chunks,
        agent_id=1,
    )

    assert stats["used_reducer"] is True
    assert stats["selected_snippets"] >= 1
    assert "16 e 18 C" in reduced
    assert "Provolone" not in reduced
    assert len(reduced) < len(chunks[0]["content"])


def test_single_agent_reducer_falls_back_to_original_when_no_snippet_matches():
    chunks = [
        {
            "content": "Texto curto sem relacao direta com a pergunta, mas ainda e melhor do que evidencia vazia.",
            "score": 0.01,
        }
    ]

    reduced, stats = _reduce_chunks_for_prompt(
        question="Como controlar estufamento tardio em queijo semiduro?",
        chunks=chunks,
        agent_id=1,
    )

    assert stats["used_reducer"] is False
    assert reduced == chunks[0]["content"]


def test_risk_question_uses_short_factual_format():
    qtype = detect_question_type(
        "Qual risco existe em usar cultura tipica de iogurte como base dominante para Parmesao?"
    )

    assert qtype == QuestionType.FACTUAL_SHORT
    assert "qual risco" in FORMAT_INSTRUCTIONS[QuestionType.FACTUAL_SHORT].lower()


def test_general_format_blocks_unasked_extra_sections():
    instruction = FORMAT_INSTRUCTIONS[QuestionType.GENERAL].lower()

    assert "1–2 parágrafos" in instruction
    assert "não acrescente recomendações" in instruction
    assert "se a pergunta não pedir" in instruction


def test_raw_milk_storage_treatment_routes_to_quality_before_regulatory():
    intent = classify_query_intent(
        "Quando o leite cru ficara armazenado por dois dias ou mais, "
        "qual tratamento e recomendado antes da pasteurizacao convencional?"
    )

    assert intent.domain == "raw_milk_process_quality"
    assert intent.search_tables[0].endswith("qualidade_leite")
    assert any(table.endswith("regulatorios") for table in intent.search_tables)
