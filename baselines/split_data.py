"""
Split ground_truth.json into train, val, and test JSON files.

Splits:
  - Training  : Inception (index 0)   → Spy (index 700)            = 701 movies
  - Validation: Lilo & Stitch (index 701) → American Gangster (850) = 150 movies
  - Testing   : Hook (index 851) → The Giver (index 1000)          = 150 movies
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    src = Path(__file__).resolve().parent.parent / "Final Annotate" / "ground_truth.json"
    out_dir = Path(__file__).resolve().parent / "splits"
    out_dir.mkdir(exist_ok=True)

    data = json.loads(src.read_text(encoding="utf-8"))
    assert len(data) == 1001, f"Expected 1001 entries, got {len(data)}"

    train = data[0:701]      # Inception → Spy
    val   = data[701:851]    # Lilo & Stitch → American Gangster
    test  = data[851:1001]   # Hook → The Giver

    print(f"Train : {len(train)} ('{train[0]['title']}' → '{train[-1]['title']}')")
    print(f"Val   : {len(val)}  ('{val[0]['title']}' → '{val[-1]['title']}')")
    print(f"Test  : {len(test)} ('{test[0]['title']}' → '{test[-1]['title']}')")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.json"
        path.write_text(
            json.dumps(split, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  → {path}")


if __name__ == "__main__":
    main()
