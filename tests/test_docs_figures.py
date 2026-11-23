"""The numbers and file names the docs state stay the ones the repo derives.

The docs quote corpus sizes and name files, scripts and links. A chunker change
or a renamed file used to leave them wrong without any test noticing. These
checks recompute what can be recomputed offline (file and chunk counts come from
the chunker, no embedding) and fail when a document says something else.
"""

import re
from pathlib import Path

import config
from rag.ingest import _chunk_file, document_paths

ROOT = Path(__file__).resolve().parent.parent
CORPORA = ("ai_act_sections", "filings_sections", "filings_statements")
# Build output the docs name but git does not track.
GENERATED = {"web/dist", "web/node_modules"}


def _markdown_docs() -> list[Path]:
    """The hand-written docs; the 308 AI Act section files are corpus text, not docs."""
    folders = [ROOT, ROOT / "docs", ROOT / "k8s"]
    docs = [p for folder in folders for p in sorted(folder.glob("*.md"))]
    return docs + sorted(ROOT.glob("data/*/README.md"))


def _sizes() -> dict[str, tuple[int, int]]:
    """(files, chunks) per corpus, as `cli.py ingest` would build them."""
    out = {}
    for corpus in CORPORA:
        paths = document_paths(corpus)
        out[corpus] = (len(paths),
                       sum(len(_chunk_file(p, corpus, config.CHUNKER)) for p in paths))
    return out


def test_readme_states_the_chunk_counts_the_chunker_produces():
    sizes = _sizes()
    sentence = (f"{sizes['ai_act_sections'][1]} AI Act chunks, "
                f"{sizes['filings_sections'][1]} 10-K text chunks and "
                f"{sizes['filings_statements'][1]} 10-K statement chunks")
    assert sentence in (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_states_the_file_counts():
    sizes = _sizes()
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{sizes['ai_act_sections'][0]} files" in text
    assert f"({sizes['filings_sections'][0]} files)" in text
    assert f"({sizes['filings_statements'][0]} files)" in text


def test_every_path_a_doc_puts_in_backticks_exists():
    pattern = re.compile(r"`((?:\.\./)?(?:api|data|deploy|docs|eval|k8s|rag|scripts|tests|web)"
                         r"/[\w./-]+)`")
    missing = []
    for doc in _markdown_docs():
        for found in pattern.findall(doc.read_text(encoding="utf-8")):
            path = found.rstrip(".,:;/")
            if path in GENERATED:
                continue
            if not any((base / path).exists() for base in (ROOT, doc.parent)):
                missing.append(f"{doc.relative_to(ROOT)}: {found}")
    assert not missing, missing


def test_every_relative_markdown_link_points_at_a_file():
    missing = []
    for doc in _markdown_docs():
        text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)
        for url in re.findall(r"\]\(([^)\s]+)\)", text):
            if url.startswith(("http", "mailto:", "#")):
                continue
            target = (doc.parent / url.partition("#")[0]).resolve()
            if not target.exists():
                missing.append(f"{doc.relative_to(ROOT)}: {url}")
    assert not missing, missing
