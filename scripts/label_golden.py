"""
Interactive golden-set labeller. Saves after every 10 rows and on quit.
Run: python label_golden.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

CSV = repo_path("golden_set", "golden_set_to_label.csv")

INTENTS = [
    "delivery_delay_or_missing",
    "wrong_or_damaged_item",
    "lost_or_stolen_package",
    "return_or_refund",
    "payment_or_gift_card_issue",
    "subscription_or_membership",
    "account_access_or_security",
    "order_status_or_tracking",
    "device_or_app_technical",
    "website_or_app_ux",
    "customer_service_quality",
    "general_complaint",
    "how_to_or_feature_question", 
]


def is_blank(v):
    return pd.isna(v) or str(v).strip() == ""


def main():
    df = pd.read_csv(CSV)
    for col in ["intent", "is_ambiguous", "is_multi_intent", "notes"]:
        df[col] = df[col].astype("object")

    start = next((i for i in range(len(df)) if is_blank(df.iloc[i]["intent"])), len(df))
    print(f"Resuming at row {start}/{len(df)}\n")
    print("Intents:")
    for i, name in enumerate(INTENTS, 1):
        print(f"  {i:2d}. {name}")
    print("Utility:")
    print("   u.  not_support_related")
    print("   i.  insufficient_context")
    print("Commands: s=skip  b=back  q=save+quit\n")

    i = start
    while i < len(df):
        row = df.iloc[i]
        print(f"--- [{i+1}/{len(df)}] stratum={row['stratum']} cluster={row.get('cluster_hint')}")
        print(f"    {str(row['text'])[:280]}")
        choice = input("  intent> ").strip().lower()

        if choice == "q":
            break
        if choice == "s":
            i += 1
            continue
        if choice == "b":
            i = max(0, i - 1)
            continue
        if choice == "u":
            df.at[i, "intent"] = "not_support_related"
        elif choice == "i":
            df.at[i, "intent"] = "insufficient_context"
        elif choice.isdigit() and 1 <= int(choice) <= len(INTENTS):
            df.at[i, "intent"] = INTENTS[int(choice) - 1]
        else:
            print("  unrecognised; try again")
            continue

        flags = input("  flags? [a=ambiguous  m=multi  am=both  enter=none] ").strip().lower()
        df.at[i, "is_ambiguous"]    = "Y" if "a" in flags else "N"
        df.at[i, "is_multi_intent"] = "Y" if "m" in flags else "N"
        df.at[i, "notes"] = input("  notes (enter to skip)> ").strip()

        i += 1
        if i % 10 == 0:
            df.to_csv(CSV, index=False)
            print(f"  [saved at row {i}]\n")

    df.to_csv(CSV, index=False)
    done = sum(1 for v in df["intent"] if not is_blank(v))
    print(f"\nSaved. Labelled {done}/{len(df)}.")


if __name__ == "__main__":
    main()