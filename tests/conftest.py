import os
import subprocess
import sys
from pathlib import Path

# run from the repo root with no real API key; tests/ on the path for fake_llm
ROOT = Path(__file__).parents[1]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ.setdefault("GOOGLE_API_KEY", "fake-key-for-tests")

# build the database and photos once (this also checks the hand labels against policy_rules.py)
subprocess.run([sys.executable, "data/make_data.py"], check=True)
