# -*- coding: utf-8 -*-
"""
agents/synthesis_rules.py — Regras canônicas de síntese do DairyApp AI.

Fonte única de verdade para:
  - R1–R9: invariantes de qualidade de resposta
  - Instruções de formato por tipo de pergunta
  - build_synthesis_prompt(): monta o HumanMessage do LLM de síntese

Usado por:
  - agents/orchestrator.py  → _build_consolidation_prompt (V1)
  - graphs/single_agent_graph.py → generate_answer (V2)

Nunca duplique estas definições em outros arquivos. Se precisar de uma
variação, adicione um parâmetro aqui — não crie um fork.
"""

from __future__ import annotations

from app.agents.orch_quality import QuestionType


# ---------------------------------------------------------------------------
# R1–R9: invariantes de qualidade — injetadas em todo prompt de síntese
# ---------------------------------------------------------------------------

SYNTHESIS_BASE_RULES: str = (
    "REGRAS DE SÍNTESE — todas obrigatórias:\n"
    "R1. Ancoragem total: use EXCLUSIVAMENTE os dados presentes nas evidências fornecidas. "
    "Nunca complete valores, doses, normas ou parâmetros com memória de treinamento.\n"
    "R2. Preservação numérica INCONDICIONAL: TODOS os valores numéricos presentes nas "
    "evidências (temperaturas, pHs, concentrações, contagens de células, limites, prazos) "
    "DEVEM aparecer na resposta. Proibido omitir qualquer número, mesmo que pareça "
    "aproximado ou que você avalie como 'não exato'. Se a evidência diz '300.000 células/mL', "
    "esse número vai na resposta — sem exceção.\n"
    "R3. Técnico e legal coexistem: recomendações técnicas e limites legais são "
    "complementares, não concorrentes. Quando os dois existem para o mesmo parâmetro, "
    "apresente AMBOS distinguindo-os claramente: 'Do ponto de vista técnico: X. "
    "O limite legal (IN 76) é Y.' Nunca descarte um pelo outro.\n"
    "R4. Completude sem excesso: responda tudo que a pergunta pede, nada além. "
    "Não adicione ressalvas genéricas, parágrafos de contexto não solicitados nem "
    "convites de continuação.\n"
    "R5. Prosa coesa: reescreva em linguagem corrida e técnica — não copie headers de seção "
    "numerados, bullets soltos, linhas iniciando no meio de uma frase nem metadados de "
    "chunk ('Trecho 1 — score 0.08').\n"
    "R6. Identidade invisível: nunca mencione agentes, bases de conhecimento, ferramentas "
    "internas, 'trechos recuperados' ou 'informações disponíveis'.\n"
    "R7. Tom: técnico, direto, profissional. Português brasileiro. Sem hedging "
    "('parece que', 'provavelmente', 'pode ser que') quando a evidência é direta.\n"
    "R8. Encerramento limpo: termine no último dado técnico relevante. "
    "Proibido: 'não foram encontradas informações adicionais', 'a norma não especifica "
    "mais detalhes', parágrafo de resumo ('Resumo:', 'Resumindo:', 'Em resumo,', "
    "'Em síntese,') — o conteúdo técnico fala por si.\n"
    "R9. Proibição de falsa ausência: NUNCA escreva 'não há especificação' ou qualquer "
    "variante de ausência quando a evidência contém um número relevante. "
    "Se o número está nas evidências, use-o — mesmo precedido de 'aproximadamente' "
    "ou 'recomenda-se'.\n"
    "R10. Valores em contexto de alerta são dados técnicos válidos: se a evidência apresenta "
    "um valor numérico no contexto de 'limite', 'excessivo', 'prejudica', 'não deve ultrapassar' "
    "ou similar, esse valor É a referência técnica para a pergunta. Use-o diretamente como "
    "resposta ao 'qual é o limite', 'qual o teor recomendado' etc. — não o descarte por estar "
    "em contexto de alerta. Exemplo: evidência diz 'sal acima de 1,8% retarda a maturação' → "
    "responda '1,8% é o limite técnico a não ultrapassar para maturação regular'."
)


# ---------------------------------------------------------------------------
# Instruções de formato por tipo de pergunta
# ---------------------------------------------------------------------------
# Estas instruções complementam o SYNTHESIS_BASE_RULES e são injetadas logo
# após as regras base no prompt de síntese, antes da evidência.

FORMAT_INSTRUCTIONS: dict[str, str] = {
    QuestionType.FACTUAL_SHORT: (
        "FORMATO DE SAÍDA: Resposta direta em 1–3 frases. "
        "O valor ou dado solicitado aparece logo na primeira frase. "
        "Sem contexto extra além do necessário para entender o dado. "
        "Se a pergunta usa 'qual risco', responda apenas o risco e o mecanismo principal; "
        "não inclua prevenção, formulação alternativa ou plano de controle, salvo se perguntado."
    ),
    QuestionType.PROCESS: (
        "FORMATO DE SAÍDA: Etapas numeradas em ordem cronológica. "
        "Cada etapa inclui os parâmetros críticos presentes nas evidências "
        "(temperatura, pH, tempo, concentração). "
        "Sem narrativa introdutória — comece diretamente na etapa 1."
    ),
    QuestionType.TROUBLESHOOTING: (
        "FORMATO DE SAÍDA obrigatório — use exatamente esta estrutura:\n\n"
        "DEFEITO: [nome em uma linha]\n\n"
        "CAUSA PROVÁVEL: [mecanismo bioquímico ou microbiológico, do mais ao menos provável]\n\n"
        "AÇÃO CORRETIVA: [intervenções concretas por ordem de impacto]\n\n"
        "Cada bloco começa com o rótulo em maiúsculas seguido de dois-pontos. "
        "Linha em branco obrigatória entre blocos. "
        "Se as evidências trouxerem parâmetro de controle preventivo, adicione: "
        "'PARÂMETRO DE CONTROLE: [valor]'. "
        "Nunca coloque DEFEITO, CAUSA e AÇÃO no mesmo parágrafo."
    ),
    QuestionType.REGULATORY: (
        "FORMATO DE SAÍDA: Cite norma, artigo e parágrafo quando presentes nas evidências. "
        "Apresente limites em formato objetivo (ex.: '≤ 300.000 células/mL — IN 76, Art. 4'). "
        "Se houver atualização ou revogação nas evidências, mencione explicitamente."
    ),
    QuestionType.COMPARATIVE: (
        "FORMATO DE SAÍDA: Tabela ou lista com eixos simétricos e explícitos. "
        "Cada critério comparado deve ter valor para ambos os lados. "
        "Diferenças mais impactantes primeiro. "
        "Linha de conclusão prática apenas se as evidências a sustentarem diretamente."
    ),
    QuestionType.CALCULATIVE: (
        "FORMATO DE SAÍDA: Fórmula → valores substituídos → resultado → unidade. "
        "Texto simples, sem LaTeX. "
        "Se houver fator de correção ou variante relevante nas evidências, liste após o resultado."
    ),
    QuestionType.GENERAL: (
        "FORMATO DE SAÍDA: Prosa técnica enxuta, normalmente 1–2 parágrafos e no máximo 6 frases. "
        "Comece pela resposta direta à pergunta. "
        "Adicione contexto técnico das evidências apenas quando for indispensável para explicar a resposta. "
        "Não acrescente recomendações, prevenção, troubleshooting, formulação alternativa, parâmetros de controle "
        "ou contexto regulatório se a pergunta não pedir isso explicitamente. "
        "Encerre no último dado técnico."
    ),
}


# ---------------------------------------------------------------------------
# build_synthesis_prompt()
# ---------------------------------------------------------------------------

def build_synthesis_prompt(
    *,
    question: str,
    question_type: str,
    specialist_text: str,
    regulatory_text: str,
) -> str:
    """Monta o HumanMessage completo para o LLM de síntese.

    Hierarquia de evidências:
      - Pergunta técnica com complemento regulatório → técnico lidera
      - Pergunta regulatória com evidência técnica   → regulatório lidera
      - Apenas técnico ou apenas regulatório         → evidência única

    Parâmetros
    ----------
    question       : texto da pergunta do usuário (já normalizado)
    question_type  : uma das constantes QuestionType
    specialist_text: evidência técnica formatada (pode ser "")
    regulatory_text: evidência regulatória formatada (pode ser "")

    Retorna
    -------
    String pronta para ser passada como HumanMessage ao LLM.
    Retorna "" quando ambas as evidências estão vazias — o chamador
    deve tratar esse caso antes de invocar o LLM.
    """
    has_specialist = bool(specialist_text.strip())
    has_regulatory = bool(regulatory_text.strip())

    if not has_specialist and not has_regulatory:
        return ""

    format_instruction = FORMAT_INSTRUCTIONS.get(
        question_type, FORMAT_INSTRUCTIONS[QuestionType.GENERAL]
    )

    is_regulatory_question = question_type == QuestionType.REGULATORY

    # ------------------------------------------------------------------
    # Bloco de evidências e instrução de síntese
    # ------------------------------------------------------------------
    if has_specialist and has_regulatory:
        if is_regulatory_question:
            evidence_block = (
                f"EVIDÊNCIA REGULATÓRIA (fonte primária):\n{regulatory_text}\n\n"
                f"EVIDÊNCIA TÉCNICA COMPLEMENTAR "
                f"(use apenas se ampliar a resposta normativa):\n{specialist_text}"
            )
            synthesis_instruction = (
                "Construa a resposta a partir da EVIDÊNCIA REGULATÓRIA como base. "
                "Use a EVIDÊNCIA TÉCNICA COMPLEMENTAR apenas para acrescentar contexto "
                "prático que a norma não cobre — não a use para sobrepor requisitos legais."
            )
        else:
            evidence_block = (
                f"EVIDÊNCIA TÉCNICA (fonte primária):\n{specialist_text}\n\n"
                f"EVIDÊNCIA REGULATÓRIA (complemento — inclua apenas o que for diretamente "
                f"relevante e não foi coberto pela evidência técnica):\n{regulatory_text}"
            )
            synthesis_instruction = (
                "Construa a resposta a partir da EVIDÊNCIA TÉCNICA como base. "
                "Inclua da EVIDÊNCIA REGULATÓRIA apenas limites, requisitos ou proibições "
                "diretamente pertinentes — nunca repita o que o técnico já cobriu. "
                "Se técnico e norma divergirem em um ponto, a norma prevalece naquele ponto."
            )

    elif has_specialist:
        evidence_block = f"EVIDÊNCIA TÉCNICA:\n{specialist_text}"
        synthesis_instruction = (
            "Construa a resposta integralmente a partir da EVIDÊNCIA TÉCNICA. "
            "Se a evidência for parcial, responda apenas a parte sustentada — "
            "não infira parâmetros ausentes."
        )

    else:  # apenas regulatório
        evidence_block = f"EVIDÊNCIA REGULATÓRIA:\n{regulatory_text}"
        synthesis_instruction = (
            "Construa a resposta integralmente a partir da EVIDÊNCIA REGULATÓRIA. "
            "Cite norma e artigo quando disponíveis. "
            "Se a evidência for parcial, responda apenas a parte sustentada — "
            "não infira requisitos não presentes."
        )

    return (
        f"PERGUNTA: {question}\n\n"
        f"{SYNTHESIS_BASE_RULES}\n\n"
        f"{format_instruction}\n\n"
        f"{evidence_block}\n\n"
        f"INSTRUÇÃO DE SÍNTESE: {synthesis_instruction}\n\n"
        f"Resposta final:"
    )
