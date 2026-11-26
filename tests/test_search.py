import subprocess
import sys

def test_hybrid_search_finds_every_labelled_question():
    out = subprocess.run([sys.executable, "search.py"], capture_output=True, text=True, check=True).stdout
    assert "hybrid  recall@3 1.00" in out, out
