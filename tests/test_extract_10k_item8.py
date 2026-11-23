"""scripts/extract_10k_item8.py flattens 10-K HTML tables into `label: v1 | v2 | v3`.

The test feeds a tiny hand-made filing (no 9 MB fixture): a hidden XBRL header
that must not leak, a table of contents that must not be taken for Item 8, a
table with "$" and ")" alignment cells, and a multi-column segment table.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import extract_10k_item8 as ex  # noqa: E402

HTML = """<DOCUMENT><TYPE>10-K<html><body>
<div style="display:none"><ix:header>hidden fact 999</ix:header></div>
<div>Item 8. Financial Statements and Supplementary Data 12</div>
<div>Item 8.&#160;&#160;Financial Statements and Supplementary Data</div>
<div>CONSOLIDATED STATEMENTS OF OPERATIONS</div>
<table>
<tr><td></td><td>September 27,2025</td><td>September 28,2024</td></tr>
<tr><td>Total net sales</td><td>$</td><td>416,161</td><td>$</td><td>391,035</td></tr>
<tr><td>Other income</td><td>(</td><td>321</td><td>)</td><td>269</td></tr>
<tr><td>Margin</td><td>46.9</td><td>%</td><td>46.2</td><td>%</td></tr>
</table>
<div>Note 1 &#8211; Policies</div><div>Text of note one.</div>
<div>Note 13 &#8211; Segment Information</div>
<table>
<tr><td>2025</td></tr>
<tr><td></td><td>Americas</td><td>Greater<br/>China</td><td>Japan</td><td>Total</td></tr>
<tr><td>Net sales</td><td>178</td><td>64</td><td>28</td><td>270</td></tr>
</table>
<div>Report of Independent Registered Public Accounting Firm</div><div>Opinion.</div>
<div>Item 9. Changes in and Disagreements with Accountants</div>
</body></html></DOCUMENT>"""


def test_hidden_header_and_toc_are_skipped():
    lines = ex.item8_lines(HTML)
    assert lines[0].startswith("Item 8.")
    assert "hidden fact 999" not in " ".join(lines)
    assert "Item 9" not in " ".join(lines)


def test_table_rows_become_label_colon_values():
    lines = ex.item8_lines(HTML)
    assert "September 27, 2025 | September 28, 2024" in lines
    assert "Total net sales: 416,161 | 391,035" in lines
    assert "Other income: (321) | 269" in lines
    assert "Margin: 46.9% | 46.2%" in lines


def test_line_break_in_a_header_cell_keeps_the_space():
    assert "Americas | Greater China | Japan | Total" in ex.item8_lines(HTML)


def test_segment_columns_are_named_on_every_value():
    out = ex.label_columns(["2025", "Americas | Greater China | Total", "Net sales: 178 | 64 | 242"])
    assert out[-1] == "Net sales (2025): Americas 178 | Greater China 64 | Total 242"


def test_shipped_statements_are_what_the_script_writes(tmp_path):
    """Running the extractor on the raw filing reproduces data/filings_statements/ exactly.

    The docs say "rerun it to regenerate them"; this is what keeps that true, and
    what stops a hand edit of a statement file from going unnoticed.
    """
    if not ex.DEFAULT_RAW.exists():
        pytest.skip("raw SEC submission not present")
    assert ex.main(["--raw", str(ex.DEFAULT_RAW), "--out", str(tmp_path)]) == 0
    written = {p.name: p.read_text(encoding="utf-8") for p in tmp_path.glob("*.txt")}
    shipped = {p.name: p.read_text(encoding="utf-8") for p in ex.DEFAULT_OUT.glob("*.txt")}
    assert written == shipped
    assert len(written) == len(ex.SECTIONS) == 19
