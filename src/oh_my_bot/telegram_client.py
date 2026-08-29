import logging
import threading
from contextlib import contextmanager

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


def send_chat_action(token: str, chat_id: int, action: str = "typing") -> None:
    # Shows the "typing..." indicator in a chat. Telegram clears it after about five seconds, so
    # anything longer has to repeat it. Failures are swallowed: this is cosmetic.
    url = f"https://api.telegram.org/bot{token}/sendChatAction"
    try:
        requests.post(url, json={"chat_id": chat_id, "action": action}, timeout=10)
    except requests.RequestException as exc:
        logger.debug("Chat action failed for chat %s: %s", chat_id, _redact(token, str(exc)))


class TypingIndicator:
    # Keeps the "typing..." indicator alive for the length of a turn, so the user gets instant
    # feedback that their message was received rather than silence until the first tool call.
    # Pausable, because while the bot is waiting for an approval tap it is waiting on the user,
    # not working, and showing "typing" there would be a lie.
    REFRESH_SECONDS = 4.0

    def __init__(self, token: str, chat_id: int):
        # Prepares an indicator for one chat; nothing is sent until start().
        self.token = token
        self.chat_id = chat_id
        self._stop = threading.Event()
        self._active = threading.Event()
        self._thread = None

    def start(self) -> None:
        # Sends the first action immediately (that is the instant feedback) and starts refreshing.
        if self._thread is not None:
            return
        self._active.set()
        send_chat_action(self.token, self.chat_id)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"typing-{self.chat_id}")
        self._thread.start()

    def _run(self) -> None:
        # Re-sends the action until stopped, skipping refreshes while paused.
        while not self._stop.wait(self.REFRESH_SECONDS):
            if self._active.is_set():
                send_chat_action(self.token, self.chat_id)

    def stop(self) -> None:
        # Ends the indicator. Safe to call more than once.
        self._stop.set()

    @contextmanager
    def paused(self):
        # Suspends the indicator for the duration of the block, e.g. while awaiting approval.
        was_active = self._active.is_set()
        self._active.clear()
        try:
            yield
        finally:
            if was_active:
                self._active.set()
                send_chat_action(self.token, self.chat_id)
