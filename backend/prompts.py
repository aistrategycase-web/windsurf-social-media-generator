# backend/prompts.py

rag_only_prompt_flexible = """
[TECNICA]
{query}

FATTO STORICO:
[ANNO]: [EVENTO]
Fonte: pagina [X]
"[CITAZIONE TESTUALE]"

DETTAGLI:
- Esecutori: [CHI]
- Data: [QUANDO]
- Luogo: [DOVE]
- Metodo: [COME]
- Prova: [DOCUMENTO/FONTE]

PRECEDENTE VERIFICATO:
[ANNO] - [EVENTO DOCUMENTATO]
Fonte: [RIFERIMENTO]

{style_guide_hashtags}
"""

rag_llm_prompt = """
Estrai un fatto storico documentato dal libro.

CONTESTO:
{book_context}

ARGOMENTO:
{query}

FORMATO:
1. FATTO: data e evento preciso
2. CITAZIONE: testo esatto e pagina
3. DETTAGLI: chi, quando, dove, come
4. PRECEDENTE: evento verificabile collegato

REGOLE:
- Solo eventi documentati
- Solo fonti verificabili
- No opinioni
- No retorica
"""