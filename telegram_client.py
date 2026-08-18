import logging

import requests

logger = logging.getLogger(__name__)


def get_updates(token: str, offset: int, timeout: int) -> list[dict]:
    # Long-polls Telegram for updates after `offset`; returns [] if there are none yet.
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {"offset": offset, "timeout": timeout}
    resp = requests.get(url, params=params, timeout=timeout + 10)
    resp.raise_for_status()
    return resp.json()["result"]


def send_message(token: str, chat_id: int, text: str) -> None:
    # Sends a text reply to a chat; logs and swallows failures since there's nothing else to do.
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to send message to chat %s", chat_id)
