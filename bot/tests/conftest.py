import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "client"))

from regime_bot.config import Settings  # noqa: E402
from regime_bot.db import Database  # noqa: E402

NOW = 1_790_000_000


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


@pytest.fixture
def settings():
    return Settings(bot_token="1:x", admin_ids=frozenset({1}), signal_secret="s3cret", referral_bonus_days=7)


class FakeBroadcaster:
    """Записывает отправленное вместо реального Telegram."""

    def __init__(self, blocked=()):
        self.sent, self.blocked, self._mid = [], set(blocked), 100

    async def send(self, chat_id, text, reply_to=None, **kw):
        if chat_id in self.blocked:
            return None
        self._mid += 1
        self.sent.append(SimpleNamespace(chat_id=chat_id, text=text, reply_to=reply_to, kw=kw, message_id=self._mid))
        return self._mid


@pytest.fixture
def bc():
    return FakeBroadcaster()


SIGNAL = {"ext_id": "btc-1", "asset": "btc/usdt", "timeframe": "4H", "side": "long", "entry": 64250, "stop": 63100,
          "targets": [66000, 67900], "confidence": 78, "regime": "бычий тренд", "rationale": "тест <b>"}
