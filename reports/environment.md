# Проверка окружения

- Windows, Python 3.13.9, 4 логических CPU, 7.84 GiB RAM.
- В начале работы было свободно 0.5–0.9 GiB RAM. Обучение использует 2 потока,
  600 пар на дату среза и 180 деревьев; валидация выполняется на полной сетке.
- Все прямые и транзитивные runtime-зависимости закреплены в requirements.txt и constraints.txt.
- Подтверждена доступность wheel-файлов NumPy 2.3.4, pandas 2.3.3,
  SciPy 1.18.1 и LightGBM 4.7.0.

## Docker: внешний блокер проверки

Установлен Docker Desktop 4.49.0. Первоначально backend падал на недоступных
AF_UNIX reparse points `Docker/run/dockerInference` и
`docker-secrets-engine/engine.sock`. Это соответствует описанию ошибки
https://github.com/docker/desktop-feedback/issues/554.

Выполнено обратимое сохранение временных runtime-папок под именами
`run.sberhack-backup-*` внутри LOCALAPPDATA/Docker и
`docker-secrets-engine.sberhack-backup-*` внутри LOCALAPPDATA.
Настройки, образы, контейнеры, тома, VHDX и зарегистрированные дистрибутивы
не удалялись и не сбрасывались. Последние папки сохранены с суффиксом 20261031021020.

После устранения сокетов backend дошёл до создания WSL VM и завершился ошибкой:

```text
Wsl/Service/RegisterDistro/CreateVm/HCS_E_CONNECTION_TIMEOUT
wsl --import-in-place docker-desktop .../Docker/wsl/main/ext4.vhdx
```

Гипервизор присутствует; WslService и vmcompute запущены; работающая vmmemWSL
при проверке отсутствовала. Попытка перезапустить WslService получила отказ
Windows: `Cannot open 'WslService' service on computer '.'`. Автоматическая
проверка разрешений инструментов действие не отклоняла; не хватает прав ОС.

Dockerfile подготовлен, но сборка и offline-проверка не могут считаться
пройденными до восстановления WSL. Не заменять этот статус на PASS по наличию Dockerfile.

## Продолжение после восстановления WSL

В PowerShell от администратора, если другие WSL-задачи закрыты:

```powershell
cd C:\Users\USER\OneDrive\Desktop\SberHack
.\tools\restart-wsl-admin.ps1
```

Если ошибка WSL сохраняется, потребуется диагностика WSL/Hyper-V на уровне Windows;
сброс Docker и удаление VHDX для решения хакатона не требуются.

После успешного `docker version`:

```powershell
docker build -t sberhack-forecast:1.0 .
docker run -d --name sberhack-offline --network none --mount "type=bind,source=C:\Users\USER\OneDrive\Desktop\SberHack\data\smoke,target=/inputs,readonly" sberhack-forecast:1.0
docker cp tools/container_smoke.py sberhack-offline:/tmp/container_smoke.py
docker exec sberhack-offline python /tmp/container_smoke.py --input-dir /inputs --network-mode none
docker image inspect sberhack-forecast:1.0 --format '{{.Size}}'
docker rm -f sberhack-offline
```

Сначала должны быть готовы `model/` и `data/smoke/forecast.csv`. Последний создаёт
`python -m tools.smoke_api --skip-prepare`. Docker smoke сравнивает все 79101
строку с CLI и проверяет кэш. Сеть контейнера отключена; доступ к CSV только через
внешний том, обучающие файлы в образ не включаются.
