from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def domain_name(path: str | Path) -> str:
    return Path(path).stem


def make_target_result_dir(
    results_root: str | Path,
    source_data: str | Path | None,
    target_data: str | Path,
) -> Path:
    source = domain_name(source_data) if source_data else "source"
    target = domain_name(target_data)
    result_dir = Path(results_root) / f"{source}_to_{target}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def save_history_csv(history: list[dict[str, float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    for row in history:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
