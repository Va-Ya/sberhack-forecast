# SberHack

Implement sales forecasting and the API in this directory. The unrelated
`android-game-tools/` directory is outside this task.

- Competition cutoff: 2026-07-26 inclusive, forecast 2026-07-27 through 2026-09-21.
- Never build features, category metadata or statistics from later observations.
- Stock duplicates are summed; negative stock is valid. Never cap sales by stock.
- Use the shared `forecast` transformations for training, CLI and API.
- Default pair universe is the full input warehouse/product cross product until
  an organizer-provided pairs CSV is supplied. Document this assumption.
- Use the four project skills in `.agents/skills`; other ECC materials are references.
- Run `.venv/Scripts/python -m pytest` and `ruff check forecast app tests tools`.
- Training: `python -m forecast.cli train --archive hakaton.zip`.
- API: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- Update HANDOFF.md and reports/experiments.jsonl after milestones. Keep the last
  working model/submission when trying another configuration.
- No Telegram messages, publication or organizer submission is authorized.
