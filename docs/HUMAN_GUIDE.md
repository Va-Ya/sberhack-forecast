# Запуск, данные и обучение

Готовый конкурсный файл — [outputs/submission.csv](../outputs/submission.csv).
Он построен на полной разрешённой истории. API создаёт отдельный прогноз по
загруженным файлам и не перезаписывает этот CSV.

## Запуск без Docker

Python 3.13. Команды из корня репозитория, PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

На Linux используйте `.venv/bin/python`. В браузере откройте
`http://127.0.0.1:8000/docs` и загрузите файлы через `POST /predict`.

`data/demo_sales.csv` и `data/demo_stock.csv` — синтетическая история одной пары за
181 день; ответ содержит 9 строк. `data/sample_*.csv` — первые 100 строк исходных
файлов для просмотра схемы. Для успешного POST их истории недостаточно.

## Входные CSV

Продажи, обязательные колонки:

```text
date,composite_division,cities_name,warehouse_name,category_lvl_1,category_lvl_2,category_lvl_3,article,characteristic,quantity,sales_sum,discount_sum
```

Остатки:

```text
date,warehouse_name,char_article,stock
```

В остатках также поддерживается `article`. Если переданы оба имени, значения
должны совпадать. Даты — `YYYY-MM-DD`, числовые поля должны читаться как числа.
Продажи неотрицательны, отрицательные остатки разрешены. Пропуски в денежных полях
допустимы: модель их не использует. Формат пустого значения — пустая ячейка CSV.
Каждый файл должен покрывать не менее 180 календарных дней.

## Обучение

Положите исходный `hakaton.zip` в корень проекта. Проверенный SHA-256:
`3fa9f5c80296328a571f80875d77dfb0b03f97859c96f59b75e86f4bcf8f3529`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -c constraints.txt
.\.venv\Scripts\python.exe -m forecast.cli train --archive hakaton.zip --iterations 400 --pairs-per-origin 1200 --threads 2
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check forecast app tests tools score.py
```

Команда обучает кандидатов на временных окнах, выбирает модель по первым двум,
оценивает на третьем и сохраняет финальную модель в `model/`, прогноз — в `outputs/`.
Параметры соответствуют текущему результату. Перед новым экспериментом сохраните
эти каталоги и `reports/selection.json` отдельно. Кэш `data/cache/` ускоряет повторный
запуск, но для свежего клона не нужен.

Обычный прогноз CLI:

```sh
python -m forecast.cli predict --sales sales.csv --stock stock.csv --output forecast.csv
```

Официальный список пар можно добавить через `--pairs allowed_pairs.csv`.
В API аналогичный файл читается из `model/allowed_pairs.csv`. Без него используется
произведение складов и товаров входных файлов; `model/pairs.csv` — каталог обучения.

## Контейнер без сети

Сначала подготовьте реальные 180-дневные входы и ожидаемый CLI-прогноз:

```sh
python -m tools.smoke_api
docker build -t sberhack-forecast:1.0 .
```

Первая команда требует локальный архив данных и зависимости тестирования.
На Linux выполните:

```sh
docker run -d --name sberhack-check --network none --mount "type=bind,source=$(pwd)/data/smoke,target=/inputs,readonly" sberhack-forecast:1.0
docker cp tools/container_smoke.py sberhack-check:/tmp/container_smoke.py
docker exec sberhack-check python /tmp/container_smoke.py --input-dir /inputs --network-mode none --extended-checks
docker stop sberhack-check
docker rm sberhack-check
```

В PowerShell замените `$(pwd)/data/smoke` на `${PWD}/data/smoke`.
Тест отправляет запросы внутри контейнера через loopback, проверяет CLI/API,
фильтры, JSON и ошибки. Последняя проведённая проверка:
[reports/docker_recovery/verification.json](../reports/docker_recovery/verification.json).

## Архив решения

```sh
python -m tools.package_solution
```

Результат — `release/sberhack-solution.zip` и манифест с контрольными суммами.
Архив содержит код, модель, прогнозы, примеры и отчёты. Собирайте его из чистого
клона: скрипт также подбирает файлы из `reports/`, где на рабочем компьютере могут
оставаться локальные дампы и логи. Проверьте содержимое ZIP перед передачей. Организаторам комплект отправляет команда.
