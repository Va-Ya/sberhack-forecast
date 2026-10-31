"""Collect current-run evidence; keep unavailable checks explicitly separate."""
from __future__ import annotations

import json
import os
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main():
    folder = Path("reports/current_audit")
    quality = json.loads((folder / "independent_quality.json").read_text())
    refit = json.loads((folder / "refit_verification.json").read_text())
    http = json.loads((folder / "http_verification.json").read_text())
    junit = ET.parse(folder / "junit.xml").getroot()
    suites = [junit] if junit.tag == "testsuite" else list(junit.iter("testsuite"))
    tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ["tests", "errors", "failures", "skipped"]}
    try:
        process = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                                 capture_output=True, text=True, timeout=20)
        docker = {"engine_available": process.returncode == 0, "exit_code": process.returncode,
                  "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        docker = {"engine_available": False, "error": str(error)}
    docker["build_and_offline_test"] = "NOT RUN"
    log = Path(os.environ.get("LOCALAPPDATA", ".")) / "Docker/log/host/com.docker.backend.exe.log"
    if log.exists():
        errors = [line for line in log.read_text(errors="replace").splitlines() if "reporting error to user:" in line]
        docker["last_startup_error"] = errors[-1] if errors else None
    metrics = pd.read_csv(folder / "independent_metrics.csv")
    control = metrics[metrics.fold == "2026-05-25"].set_index("candidate")
    read_checks = [c for c in http["checks"] if c["check"] in
                   {"full_cached_csv", "full_json", "article_filter", "warehouse_filter", "pair_filter"}]
    result = {"time_utc": datetime.now(timezone.utc).isoformat(), "unit_tests": tests,
              "http_checks": len(http["checks"]), "http_all_passed": all(c["passed"] for c in http["checks"]),
              "real_post_seconds": [c["seconds"] for c in http["checks"] if c["check"].startswith("real_180")],
              "get_seconds": {c["check"]: c["seconds"] for c in read_checks},
              "source_audit_passed": quality["status"] == "passed",
              "refit_byte_identical": refit["native_model_bytes_equal"],
              "refit_seconds": refit["elapsed_seconds"], "docker": docker,
              "production_fix": "Explicit missing/directory --pairs path now exits with code 2; no silent Cartesian fallback",
              "main_model_replaced": False, "competition_truth_available": False}
    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Свежая проверка SberHack — 31.10.2026", "",
             "Проверялись текущие файлы и работающий HTTP-сервер. Прежний отчёт не использовался как доказательство: "
             "данные и метрики пересчитаны из исходного ZIP.", "",
             f"- Тесты: {tests['tests']}; ошибок {tests['errors']}, падений {tests['failures']}, пропусков {tests['skipped']}.",
             f"- Реальный HTTP API: {len(http['checks'])} проверок; все успешны: {result['http_all_passed']}.",
             "- Повторные POST на реальных 180-дневных CSV: " +
             ", ".join(f"{v:.2f} с" for v in result["real_post_seconds"]) + "; по 79 101 строке.",
             "- GET полного CSV/JSON и фильтров: " +
             ", ".join(f"{k}: {v:.3f} с" for k, v in result["get_seconds"].items()) + ".",
             "- Кэш после ошибочного POST сохраняется; повторный POST даёт точно одинаковые байты; CSV/JSON согласованы.",
             f"- Повторное финальное обучение: {refit['elapsed_seconds']:.2f} с; файл модели побайтово совпал: "
             f"{refit['native_model_bytes_equal']}; расхождение прогнозов {refit['max_abs_prediction_difference']}.",
             "- Обучение повторено только для одной выбранной финальной модели на том же Windows-хосте; "
             "межплатформенная идентичность не проверена.", "", "## Качество: независимое контрольное окно", "",
             "Факты 25.05–26.07.2026; модель обучена по 24.05.2026. Меньше — лучше.", "",
             "| Прогноз | SMAPE | WAPE | RMSE | Bias | RMSLE |", "|---|---:|---:|---:|---:|---:|"]
    for name in ["last", "mean4", "mean8", "seasonal", "log_zero", "log_zero_180day_history"]:
        row = control.loc[name]
        lines.append("| " + name + " | " + " | ".join(f"{row[k]:.4f}" for k in
                     ["smape", "wape", "rmse", "bias", "rmsle"]) + " |")
    fold = quality["folds"]["2026-05-25"]
    lines += ["", "## Ограничения качества", "",
              f"- Суммарно прогноз {fold['predicted_total']:.2f} при факте {fold['actual_total']:.2f}: "
              f"систематическое занижение {abs(fold['signed_total_bias_percent']):.2f}%.",
              "- Показатель Bias в таблице — среднее по сериям; он отличается от общего знакового смещения выше.",
              "- Простое среднее за 8 недель лучше выбранной модели по RMSE, Bias и немного по RMSLE. "
              "Выбранная модель выигрывает по совокупности метрик с большим весом SMAPE; она не лучшая по каждой метрике.",
              "- На 180 днях WAPE немного хуже полной истории: 47.4211% против 47.3994%. "
              "Строгое равенство качества при сокращении истории не подтверждено.",
              "- На редких сериях ошибки существенно выше; см. slice_metrics.csv. SMAPE не является процентом точности.",
              "- Фактов для всех конкурсных недель 27.07–27.09.2026 нет. Реальную конкурсную ошибку вычислить нельзя.",
              "- Сетка 11 × 799 остаётся допущением: эталон пар не предоставлен. В исторических окнах используется "
              "один фиксированный каталог пар из всей разрешённой истории.", "", "## Данные и воспроизводимость", "",
              "- Из ZIP независимо восстановлены 1 492 370 строк продаж и 6 153 988 строк остатков.",
              "- Значения панелей продаж/остатков совпали точно, максимальная разница 0.0; маски отсутствующих остатков совпали.",
              "- Суммирование 1 250 375 повторных строк остатков внутри панели подтверждено; отрицательные остатки сохранены.",
              "- Все три замороженные модели повторно рассчитали прогнозы; метрики другой реализацией совпали с прежними.",
              "- Данные после 26.07 исключены: 13 543 строки продаж и 61 683 строки остатков.", "",
              "## Найденный и исправленный дефект CLI", "",
              "Явно указанный отсутствующий файл --pairs раньше молча игнорировался; каталог вместо файла вызывал "
              "необработанное исключение. Теперь оба случая завершаются понятной ошибкой CLI с кодом 2. "
              "Две регрессионные проверки сначала воспроизвели ошибку, затем вошли в финальный прогон. "
              "Рабочая модель и конкурсный CSV не заменялись.", "", "## Docker", "",
              f"Engine доступен сейчас: {docker['engine_available']}. Сборка и автономный тест: **NOT RUN**.",
              "Последняя свежая ошибка запуска:", "```text", str(docker.get("last_startup_error", docker.get("stderr", ""))),
              "```", "Docker Desktop не может пересоздать временный dockerInference. Это проверка окружения; "
              "контейнерная работоспособность и лимиты размера/сборки пока не доказаны.", "",
              "Полные данные проверки: summary.json, independent_quality.json, independent_metrics.csv, "
              "horizon_metrics.csv, slice_metrics.csv, refit_verification.json, http_verification.json и junit.xml.", ""]
    (folder / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
