"""Generate an auditable validation report from saved out-of-time predictions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.metrics import compute_metrics


def main():
    report_dir = Path("reports")
    summary = json.loads((report_dir / "training_summary.json").read_text())
    selected = summary["selected"]
    lines = ["# Отчёт валидации", "", f"Выбранный кандидат: **{selected}**.", "",
             "Выбор зафиксирован по окнам 19.01 и 23.03.2026 до оценки окна 25.05.2026.",
             "Каждое окно содержит 9 недель. Все метрики рассчитаны на полной сетке пар.", "",
             "| Начало окна | Роль | SMAPE | WAPE | RMSE | Bias | RMSLE |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for i, (fold, metrics) in enumerate(summary["folds"].items()):
        for name in ["last", selected]:
            m = metrics[name]
            role = ("контроль" if i == 2 else "выбор") + " / " + name
            lines.append(f"| {fold} | {role} | " + " | ".join(f"{m[k]:.4f}" for k in
                         ["smape", "wape", "rmse", "bias", "rmsle"]) + " |")
    lines += ["", "Веса: SMAPE 50%, WAPE 20%, RMSE 15%, |Bias| 10%, RMSLE 5%.",
              "Ранговые результаты относятся к фиксированному набору локальных кандидатов, не к лидерборду конкурса.",
              "", "## Контрольное окно: разбор ошибок", ""]
    with np.load(report_dir / "predictions/2026-05-25.npz") as arrays:
        y, p = arrays["truth"], arrays[selected]
    masks = {"нулевые за всё окно": y.sum(axis=0) == 0,
             "редкие, не менее половины недель нулевые": ((y == 0).mean(axis=0) >= 0.5) & (y.sum(axis=0) > 0),
             "регулярные": (y == 0).mean(axis=0) < 0.5}
    slices = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        m = compute_metrics(y[:, mask], p[:, mask])
        slices[name] = {"series": int(mask.sum()), **m}
        lines.append(f"- {name}: {mask.sum()} серий; SMAPE {m['smape']:.2f}, WAPE {m['wape']:.2f}%.")
    pairs = pd.read_csv("model/pairs.csv", dtype=str)
    errors = pairs.copy()
    errors["actual_total"] = y.sum(axis=0)
    errors["predicted_total"] = p.sum(axis=0)
    errors["absolute_error"] = np.abs(y - p).sum(axis=0)
    errors.sort_values("absolute_error", ascending=False).head(30).to_csv(report_dir / "largest_errors.csv", index=False)
    lines += ["", "Наибольшие абсолютные ошибки сохранены в largest_errors.csv.",
              "", "## Воспроизводимость и ограничения", "",
              f"- Повторный инференс совпал точно: {summary['deterministic_repeat']}.",
              f"- Строк submission: {summary['rows']}.",
              f"- SHA-256 submission: `{summary['submission_sha256']}`.",
              f"- Время команды обучения: {summary['elapsed_seconds']:.1f} секунд.",
              "- Состав пар временно принят равным полному произведению 11 × 799; эталон организаторов отсутствует.",
              "- Конкурсная отсечка 26.07.2026; горизонт 27.07–21.09.2026.",
              "- Контрольные факты находятся до отсечки; конкурсных будущих фактов для оценки нет.",
              f"- Обучение: {summary['config']}",
              "- Качество на новом распределении и при сокращённой истории не гарантируется.", ""]
    (report_dir / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "error_slices.json").write_text(json.dumps(slices, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Validation report ready", flush=True)


if __name__ == "__main__":
    main()
