import logging

import requests

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
EMPTY_REPLY_FALLBACK = "Sorry, I got an empty response."


def _redact(token: str, text: str) -> str:
    # Removes the bot token from a string so it never lands in logs (Telegram embeds it in the URL path).
    return text.replace(token, "<redacted>")


def get_updates(token: str, offset: int, timeout: int) -> list[dict]:
    # Long-polls Telegram for updates after `offset`; returns [] if there are none yet.
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {"offset": offset, "timeout": timeout}
    resp = requests.get(url, params=params, timeout=timeout + 10)
    resp.raise_for_status()
    return resp.json()["result"]


def _send_single_message(token: str, chat_id: int, text: str, reply_markup=None) -> None:
    # Sends one chunk of text to a chat via a single sendMessage call, optionally with an
    # inline keyboard attached.
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()


def send_message(token: str, chat_id: int, text: str, reply_markup=None) -> None:
    # Sends a text reply to a chat, splitting it into <=4096-char chunks (Telegram's limit) and
    # substituting a fallback string for empty/None content. Any inline keyboard goes on the final
    # chunk only, so the buttons sit under the end of the message. Logs and swallows failures
    # (with the bot token redacted) since there's nothing else to do.
    if not text:
        text = EMPTY_REPLY_FALLBACK
    chunks = [
        text[i : i + TELEGRAM_MAX_MESSAGE_LENGTH]
        for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH)
    ] or [text]
    try:
        for index, chunk in enumerate(chunks):
            markup = reply_markup if index == len(chunks) - 1 else None
            _send_single_message(token, chat_id, chunk, markup)
    except requests.RequestException as exc:
        logger.error("Failed to send message to chat %s: %s", chat_id, _redact(token, str(exc)))


def answer_callback_query(token: str, query_id: str, text: str = "") -> None:
    # Acknowledges a tapped inline button so Telegram stops showing the spinner on it.
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    try:
        resp = requests.post(url, json={"callback_query_id": query_id, "text": text}, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to answer callback query: %s", _redact(token, str(exc)))
