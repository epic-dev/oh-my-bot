import logging
import multiprocessing
import queue
import threading

from llm_client import build_messages
from telegram_client import send_message

logger = logging.getLogger(__name__)


class ChatLocks:
    def __init__(self):
        # Holds one Lock per chat_id, created on first use, so each user's messages are serialized.
        self._locks: dict[int, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def get(self, chat_id: int) -> threading.Lock:
        # Returns the Lock for this chat_id, creating it the first time it's requested.
        with self._registry_lock:
            if chat_id not in self._locks:
                self._locks[chat_id] = threading.Lock()
            return self._locks[chat_id]


def _llm_call_target(connector, messages, result_queue):
    # Runs inside the child process: calls the connector and puts the outcome on the queue.
    try:
        result_queue.put(("ok", connector.complete(messages)))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def run_llm_call(connector, messages: list[dict], timeout: int) -> str:
    # Runs connector.complete in its own process; kills it and returns an error string on timeout.
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_llm_call_target, args=(connector, messages, result_queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return "Sorry, that took too long. Please try again."
    try:
        status, payload = result_queue.get(timeout=1)
    except queue.Empty:
        return "Sorry, I couldn't reach the AI service. Please try again shortly."
    if status == "error":
        logger.error("LLM call failed: %s", payload)
        return "Sorry, I couldn't reach the AI service. Please try again shortly."
    return payload


def handle_update(update, config, connector, chat_locks, telegram_token):
    # Processes one Telegram update end-to-end: serialize per chat, call the LLM, send the reply.
    # The whole pipeline (chat_id/text extraction through the final send_message) is guarded so
    # no unexpected exception ever escapes this function.
    message = update.get("message")
    if not message or "text" not in message:
        return
    chat_id = None
    try:
        chat_id = message["chat"]["id"]
        text = message["text"]
        lock = chat_locks.get(chat_id)
        with lock:
            messages = build_messages(chat_id, text)
            reply = run_llm_call(connector, messages, config.llm_timeout_seconds)
            send_message(telegram_token, chat_id, reply)
    except Exception:
        logger.exception("Unexpected error handling update for chat %s", chat_id)
        if chat_id is not None:
            send_message(telegram_token, chat_id, "Sorry, something went wrong. Please try again.")
