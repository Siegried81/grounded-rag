"""Strict scoring rules: numbers at the displayed precision, cited refusals, more languages.

Pure functions only, no network and no LLM call.
"""

from __future__ import annotations

import pytest

from rag.answer_metrics import (
    is_cited_refusal,
    is_refusal,
    key_fact_present,
    score_answer,
    summarize,
)
from rag.types import Chunk, Retrieved

SOURCES = [Retrieved(Chunk("c1", "The fine is 35 million euros.", "doc1.txt", 1, "corp"), 0.9)]


# --- numbers at the displayed precision --------------------------------------

@pytest.mark.parametrize("fact, answer, expected", [
    # The cases decided with the user.
    ("15,421", "Total: 15 413", False),
    ("7,979 million", "about 8,0 billion", True),
    ("6.4%", "up 6 %", True),
    ("46.9%", "margin of 46.9 %", True),
    ("46,9 %", "margin of 46.9 %", True),
    ("$416,161 million", "Net sales were 416,161 million", True),
    # Rounding to the precision shown, no tolerance beyond it.
    ("7,979 million", "8 billion", True),
    ("7,979 million", "7.98 billion", True),
    ("7,979 million", "7.9 billion", False),
    ("7,979 million", "7.97 billion", False),  # inside 1 %, but not what 7,979 rounds to
    ("166,000", "about 167,000", False),
    ("166,000", "about 166 000", True),
    # An answer more precise than the expected value is not matched.
    ("46.9%", "46.93 %", False),
    ("46.9%", "46.90 %", True),
    ("8.0 billion", "7,979 million", False),
    # Half rounds up.
    ("2.5", "3", True),
    ("2.5", "2", False),
    ("6.45%", "6.5%", True),
    ("6.45%", "6.4%", False),
    ("2.675", "2.68", True),
    # Scales and unscaled table copies.
    ("€500 million", "0.5 billion euros", True),
    ("€500 million", "500 billion euros", False),
    ("$416,161 million", "Total net sales $ 416,161 (in millions)", True),
    # A percent fact still needs a percent.
    ("10%", "about 10 units", False),
])
def test_numeric_fact_at_displayed_precision(fact, answer, expected):
    assert key_fact_present(fact, answer) is expected


# --- cited refusals ----------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The sources do not mention the fine [S1].",
    "Les sources fournies ne précisent pas le montant des amendes [S2].",
    "I do not know. The sources describe the general regime for providers [S1].",
    "I do not know which models were designated, as the provided sources describe the "
    "procedures but do not name any particular model. [S1][S2][S3]",
    "I'm sorry, but the provided sources do not include a specific percentage for the "
    "market share. [S1][S3]",
    "I do not know based on the sources [S1]. The documents only discuss obligations [S2].",
    "Je ne sais pas [S1]. Les passages portent sur les obligations des fournisseurs [S2].",
    "The sources do not mention the fine [S1]. The context does not specify a figure [S2].",
])
def test_a_cited_refusal_is_a_refusal(text):
    assert is_cited_refusal(text)
    assert is_refusal(text)


@pytest.mark.parametrize("text", [
    "I don't know for sure, but the fine is 35 million [S1].",
    "Je ne sais pas exactement, mais l'amende est de 35 millions [S1].",
    "I do not know. The fine is 35 million [S1].",
    "I do not know. The sources say the fine is 35 million [S1].",
    "The sources do not mention X [S1]. The fine is 35 million [S2].",
    "Le report vise les systèmes Annexe III [S1]. Les sources ne précisent pas le reste.",
    "Apple has five segments [S1].",
])
def test_a_hedged_claim_is_not_a_refusal(text):
    assert not is_cited_refusal(text)
    assert not is_refusal(text)


def test_cited_refusal_is_flagged_and_counted():
    item = {"id": "u-01", "answerable": False, "key_facts": []}
    row = score_answer(item, "The sources do not mention the fine [S1].", False, SOURCES)
    assert row["refused"] and row["cited_refusal"] and row["grounding_score"] is None
    plain = score_answer(item, "I do not know.", False, SOURCES)
    assert plain["refused"] and not plain["cited_refusal"]
    gate = score_answer(item, "The sources do not mention the fine [S1].", True, SOURCES)
    assert gate["refused"] and not gate["cited_refusal"]
    s = summarize([row, plain, gate])
    assert s["n_refused"] == 3 and s["n_cited_refused"] == 1


# --- languages ---------------------------------------------------------------

@pytest.mark.parametrize("text", [
    # Dutch
    "Ik weet het niet.",
    "Ik kan deze vraag niet beantwoorden op basis van de bronnen.",
    "Ik kan u hier niet mee helpen.",
    "Er is geen informatie over de boete in de bronnen.",
    "De bronnen bevatten geen informatie over dit onderwerp.",
    "De bronnen vermelden het bedrag niet.",
    "Het bedrag wordt niet vermeld in de documenten.",
    "De bronnen bevatten hierover niets [S1].",
    # Spanish
    "No dispongo de la información necesaria para responder a esa pregunta.",
    "No puedo responder a esa pregunta.",
    "No tengo información sobre eso.",
    "Las fuentes no contienen información sobre la multa.",
    "Las fuentes no mencionan el importe.",
    "La información de las fuentes es insuficiente.",
    # German
    "Ich weiß nicht.",
    "Ich kann diese Frage nicht beantworten.",
    "Dazu habe ich keine Informationen.",
    "Die Quellen enthalten keine Angaben zur Geldbuße.",
    "Die Quellen erwähnen den Betrag nicht.",
    "Der Betrag wird in den Dokumenten nicht erwähnt.",
    "Ich kann Ihnen dabei nicht helfen.",
])
def test_refusals_in_dutch_spanish_german(text):
    assert is_refusal(text)


@pytest.mark.parametrize("text", [
    # Dutch content
    "De boete bedraagt maximaal 35 miljoen euro [S1].",
    "Aanbieders moeten een risicobeheersysteem opzetten en documenteren [S2].",
    "Apple heeft vijf segmenten: Amerika, Europa, Groot-China, Japan en de rest van Azië [S1].",
    "De verordening is niet van toepassing op militaire systemen [S1].",
    # Spanish content
    "La multa máxima es de 35 millones de euros [S1].",
    "Los proveedores deben establecer un sistema de gestión de riesgos [S2].",
    "El Reglamento no se aplica a los sistemas militares [S1].",
    # German content
    "Die Geldbuße beträgt höchstens 35 Millionen Euro [S1].",
    "Anbieter müssen ein Risikomanagementsystem einrichten [S2].",
    "Die Verordnung gilt nicht für militärische Systeme [S1].",
    "Die Quellen nennen eine Geldbuße von 35 Millionen Euro [S1].",
])
def test_content_in_dutch_spanish_german_is_not_a_refusal(text):
    assert not is_refusal(text)
