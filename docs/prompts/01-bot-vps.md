# Промт 01 — Запуск @regime_ai_bot на VPS

Скопируйте всё ниже в Claude Code на своём компьютере.

---

Ты — DevOps-инженер. Помоги мне запустить в продакшен Telegram-бота `@regime_ai_bot`, код которого уже написан
и покрыт тестами: репозиторий `https://github.com/bitglumerzz/market-regime-dashboard`, ветка
`claude/modest-tesla-bcavni`, папка `bot/`. Сначала прочитай `bot/README.md`, `bot/.env.example`,
`bot/regime_bot/__main__.py` и `bot/regime_bot/api.py`, затем действуй по плану. Я выполняю команды на сервере
сам, если у тебя нет к нему доступа — тогда давай мне их по одной группе и жди вывода.

**Что уже есть в боте:** aiogram 3.31, оплата Telegram Stars (месяц с автопродлением, 3 и 12 месяцев),
согласие на ПДн, дисклеймер, подписки в SQLite, рефералы, метки кампаний `?start=src_…`, напоминания,
админ-команды, HTTP API `POST /api/v1/signals` с подписью HMAC для моей локальной модели. Режимы MODE=polling/webhook.

**Цель:** бот работает 24/7 на VPS вне РФ, принимает оплату, API сигналов доступен только по HTTPS.

## Шаги
1. **Сервер.** Помоги выбрать и заказать VPS: рекомендация исследования — Hetzner CX23 (≈€5.49/мес,
   Falkenstein/Helsinki). Ubuntu 24.04, вход по SSH-ключу, отключить вход по паролю и root, `ufw`
   (открыты только 22, 80, 443), `unattended-upgrades`, fail2ban.
2. **Домен.** Мне нужен поддомен вида `bot.<мой-домен>` с A-записью на IP сервера. Если домена нет —
   предложи дешёвый вариант и проведи через покупку и DNS.
3. **Docker.** Установи Docker Engine + compose plugin. Склонируй репозиторий в `/opt/regime`, ветка
   `claude/modest-tesla-bcavni`.
4. **Секреты.** Создай `bot/.env` из `.env.example`. `BOT_TOKEN` я вставлю сам (не проси прислать его в чат).
   Сгенерируй `SIGNAL_API_SECRET` и `WEBHOOK_SECRET` командой `python3 -c "import secrets;print(secrets.token_hex(32))"`,
   сохрани их в `.env` и скажи мне сохранить `SIGNAL_API_SECRET` в менеджер паролей — он понадобится на Mac.
   `ADMIN_IDS` — мой Telegram ID (подскажи, как узнать). `MODE=webhook`, `WEBHOOK_BASE=https://bot.<домен>`,
   `API_HOST=0.0.0.0`, `API_PORT=8081`.
5. **HTTPS.** Caddy как reverse proxy: `bot.<домен>` → `localhost:8081`, автоматический Let's Encrypt.
   Проксировать только пути `/tg/webhook`, `/api/v1/*`, `/healthz`; остальное 404. Порт 8081 наружу
   НЕ публиковать (в `docker-compose.yml` поменяй `"8081:8081"` на `"127.0.0.1:8081:8081"`).
6. **Запуск.** `docker compose up -d --build` в `bot/`. Проверь логи, `curl https://bot.<домен>/healthz`,
   `getWebhookInfo` (без ошибок, `pending_update_count` не растёт).
7. **Проверка оплаты.** Пройди со мной сценарий: /start → Тарифы → согласие → риски → купить «3 месяца».
   Для теста временно поставь в `PLANS_JSON` тариф за 1 ⭐, оплати, проверь «Мой доступ», `/stats` у админа,
   затем сделай `/refund <charge_id>` и верни нормальные цены.
8. **Проверка API сигналов.** С моего компьютера: `bot/client/push_signal.py send` с тестовым JSON
   (ext_id `test-1`), убедись, что сигнал пришёл мне в Telegram (у меня должен быть доступ — выдай `/grant <мой id> 1`),
   затем `close test-1 cancelled`.
9. **Надёжность.** Ежедневный бэкап `bot/data/regime_bot.sqlite3` (sqlite3 `.backup`) + `restic` в Backblaze B2
   или другой S3, хранение 30 дней; проверка восстановления. Uptime Kuma (или Healthchecks.io) на `/healthz`
   с оповещением мне в Telegram. `docker compose` с `restart: unless-stopped` и logrotate для логов Docker.
10. **Документация.** Допиши в `bot/README.md` раздел «Продакшен» с фактическими шагами и командами
    обновления (`git pull && docker compose up -d --build`), закоммить в ветку и запушь.

## Ограничения
- Не меняй бизнес-логику бота без моего согласия. Если нашёл баг — опиши, предложи фикс, прогони
  `python -m pytest -q` в `bot/` до и после.
- Никогда не печатай в чат токен бота, секреты и содержимое `.env`.
- Не открывай наружу SQLite, порт 8081 и SSH по паролю.

## Готово, когда
- `https://bot.<домен>/healthz` → `{"ok": true}`, вебхук без ошибок, бот отвечает в Telegram.
- Тестовая оплата прошла и возвращена, тестовый сигнал доставлен и закрыт.
- Бэкап и мониторинг настроены и проверены; README обновлён и запушен.
