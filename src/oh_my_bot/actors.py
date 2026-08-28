import logging
import queue
import threading

logger = logging.getLogger(__name__)


class ActorPool:
    def __init__(self, handler):
        # Owns one long-lived thread and queue per chat, each running `handler` over its items.
        # This replaces the thread pool: a chat blocked waiting for a user to tap an approval
        # button now holds only its own thread, so it cannot delay any other chat.
        self._handler = handler
        self._queues = {}
        self._lock = threading.Lock()

    def submit(self, chat_id: int, item) -> None:
        # Appends one item to a chat's queue, starting that chat's actor if this is its first.
        self._queue_for(chat_id).put(item)

    def _queue_for(self, chat_id: int) -> queue.Queue:
        # Returns the chat's queue, creating the queue and its actor thread on first use.
        with self._lock:
            existing = self._queues.get(chat_id)
            if existing is not None:
                return existing
            chat_queue = queue.Queue()
            self._queues[chat_id] = chat_queue
        thread = threading.Thread(
            target=self._run, args=(chat_id, chat_queue), daemon=True, name=f"actor-{chat_id}"
        )
        thread.start()
        return chat_queue

    def _run(self, chat_id: int, chat_queue: queue.Queue) -> None:
        # The actor loop: drain this chat's queue forever, strictly in order. Every exception is
        # caught so one failed turn never kills the actor and strands the chat.
        while True:
            item = chat_queue.get()
            try:
                self._handler(item)
            except Exception:
                logger.exception("Actor for chat %s failed on an item", chat_id)

    def active_chats(self) -> int:
        # Reports how many chats currently have an actor, for /status and diagnostics.
        with self._lock:
            return len(self._queues)
