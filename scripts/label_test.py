import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

df = pd.read_csv(repo_path("golden_set", "golden_set_to_label.csv"))

# Random 20 general_complaint rows to eyeball
gc = df[df["intent"] == "general_complaint"].sample(20, random_state=42)
for _, r in gc.iterrows():
    print(f"[{r['conversation_id']}] {str(r['text'])[:180]}")
    print(f"   notes: {r['notes']}")
    print()