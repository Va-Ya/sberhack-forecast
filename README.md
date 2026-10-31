# SberHack — прогноз продаж

Текущий опубликованный снимок: активна модель `log_t0.75` (400 деревьев).
Новое исследование ансамблей сохранено в `reports/research/`, но его итоговое
обучение остановлено по запросу пользователя; ансамбль не заменяет активную модель.
Дополнительные тесты новой исследовательской версии не запускались по запросу пользователя.
Прирост 20% по каждой метрике не достигнут. Подробности: `docs/PUBLICATION_STATUS.md`.

LightGBM: log1p(quantity), 400 деревьев, 65 признаков, порог обнуления 0.75.
Горизонт: 9 недель с 27.07.2026; история ограничена 26.07.2026 включительно.
Сводка изменений и сравнение всех пяти метрик: [reports/AUDIT.md](reports/AUDIT.md).
Полное ТЗ сохранено в [docs/SPECIFICATION.md](docs/SPECIFICATION.md).

## Сборка

Docker Desktop с работающим Linux Engine либо Docker Engine, linux/amd64. GPU не требуется.
Модель уже обучена и находится в model/. Образ не содержит исходные обучающие CSV.

```sh
docker build -t sberhack-forecast:improved .
docker run -d --name sberhack-api -p 127.0.0.1:8000:8000 sberhack-forecast:improved
```

Интерактивный интерфейс: http://127.0.0.1:8000/docs .
Обычный запуск публикует API на localhost; отдельная проверка с --network none
подтверждает, что самой модели не нужны внешние сетевые обращения.

## Локальное окружение и воспроизведение

Проверено локально Python 3.12; контейнер использует Python 3.13.9.

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt -c constraints.txt
.venv/Scripts/python -m forecast.cli train --archive hakaton.zip --iterations 400 --pairs-per-origin 1200 --threads 2
.venv/Scripts/python -m pytest
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Для другого эксперимента используйте отдельную копию проекта: несовпадение кода или
параметров с сохранённым фолдом намеренно останавливает повторное использование кэша.

## API

```sh
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -F "sales=@sales.csv" -F "stock=@stock.csv" -o forecast.csv
curl "http://127.0.0.1:8000/predict?article=char_article_1" -o article.csv
curl "http://127.0.0.1:8000/predict?warehouse=warehouse_name_1" -o warehouse.csv
curl "http://127.0.0.1:8000/predict?article=char_article_1&warehouse=warehouse_name_1" -o pair.csv
curl "http://127.0.0.1:8000/predict/all?format=csv" -o all.csv
curl "http://127.0.0.1:8000/predict/all?format=json" -o all.json
```

В PowerShell используйте curl.exe. POST принимает sales и stock с историей минимум
180 дней. Прогноз начинается в понедельник следующей календарной недели после max(sales.date).
Остатки после этой даты отбрасываются. Дубли остатков суммируются, отрицательные допустимы;
продажи остатками не ограничиваются. GET доступен после успешного POST; кэш в памяти
до перезапуска. Неверный POST сохраняет предыдущий прогноз.
Ошибки: 400 — пустой/нечитаемый CSV или неверные параметры; 422 — схема/история/значения;
409 — нет прогноза либо параллельный расчёт; 404 — неизвестная пара; 500 — ошибка модели.
Ответ ошибки содержит detail и code. Одновременно выполняется один POST.

## Структура

```text
forecast/  данные, признаки, модель, метрики, CLI
app/       FastAPI, общий предиктор
model/     обученная модель, признаки и кодировки
outputs/   submission.csv и наивные прогнозы
reports/   аудит, метрики, журнал и результаты проверок
tests/     контрактные и регрессионные тесты
tools/     проверка API/контейнера и упаковка
docs/      ТЗ и инструкции
```

## Ограничения

Официальный список пар не предоставлен: используется полная сетка 11×799.
Если появится список warehouse_name,article, передайте --pairs при обучении либо
используйте model/allowed_pairs.csv для API. Структуру выхода всегда нужно сверять с эталоном.
На 180 днях истории годовые признаки недоступны. Качество на скрытом тесте неизвестно.
Исходные hakaton.zip, окружение и кэши не входят в поставку; для переобучения нужен архив.
Состояние проверок находится в HANDOFF.md и reports/. Старые инструкции docs/HUMAN_GUIDE.md
и docs/AGENT_GUIDE.md содержат сведения исходного выпуска; текущие команды приведены здесь.
