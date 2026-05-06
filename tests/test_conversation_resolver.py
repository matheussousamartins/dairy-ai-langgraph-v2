from app.rag.conversation_resolver import resolve_conversation_turn
from app.rag.conversation_resolver import should_use_conversation_context
from app.server.webapp import _build_orchestrator_input_messages


HISTORY = [
    {
        "role": "human",
        "content": "Qual contagem de celulas somaticas e recomendada para leite destinado a Parmesao?",
    },
    {
        "role": "ai",
        "content": (
            "Para Parmesao, recomenda-se CCS nao superior a 300.000 celulas por mL; "
            "legalmente, a IN 76 estabelece 500.000 CS/mL para leite cru refrigerado."
        ),
    },
]


def test_short_regulatory_followup_depends_on_previous_turn():
    resolution = resolve_conversation_turn("e do ponto de vista da legislacao?", HISTORY)

    assert resolution.depends_on_previous
    assert resolution.intent == "followup"


def test_deepening_request_uses_history_even_when_not_tiny():
    resolution = resolve_conversation_turn(
        "Agora explique melhor o impacto tecnologico dessa CCS elevada no rendimento e na maturacao.",
        HISTORY,
    )

    assert resolution.depends_on_previous
    assert resolution.intent in {"deepening", "followup"}


def test_standalone_topic_shift_does_not_inject_history():
    assert not should_use_conversation_context(
        "Qual o limite de coliformes para queijo minas frescal?",
        HISTORY,
    )


def test_orchestrator_message_gets_context_for_deepening_request():
    messages = _build_orchestrator_input_messages(
        "session-1",
        "Explique melhor esse limite e o impacto tecnologico.",
        None,
        preloaded_history=HISTORY,
    )

    assert len(messages) == 1
    content = messages[0].content
    assert "[Contexto recente da conversa]" in content
    assert "[Pergunta atual]" in content
    assert "300.000" in content
    assert "Explique melhor esse limite" in content
