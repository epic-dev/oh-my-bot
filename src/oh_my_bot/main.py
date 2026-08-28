import logging
import time
from concurrent.futures import ThreadPoolExecutor

from .config import load_config
from .llm_client import OpenAICompatConnector
from .store import Store
from .telegram_client import _redact, get_updates
from .worker import ChatLocks, handle_update

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def should_handle(update, allowed_user_ids) -> bool:
    # Decides whether one update may be processed: it must be a text message, from an allowlisted
    # user, in a private chat. The allowlist keys on the *user* id, never the chat id — in a group
    # those differ, and a chat-id check would authorize every member of an allowed group.
    message = update.get("message")
    if not message or "text" not in message:
        return False
    user_id = (message.get("from") or {}).get("id")
    chat_type = (message.get("chat") or {}).get("type")
    if user_id not in allowed_user_ids or chat_type != "private":
        logger.info("Dropping update from user %s (chat type %s)", user_id, chat_type)
        return False
    return True


def main():
    # Loads config, starts the thread pool, and long-polls Telegram forever, dispatching updates to workers.
    config = load_config()
    connector = OpenAICompatConnector(
        config.llm_base_url,
        config.llm_model,
        config.reasoning_tags,
        config.stop_sequences,
        config.llm_max_tokens,
    )
    store = Store(config.db_path)
    store.init_schema()
    chat_locks = ChatLocks()
    offset = 0
    backoff = 1

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        while True:
            try:
                updates = get_updates(config.telegram_bot_token, offset, config.poll_timeout_seconds)
                backoff = 1
            except Exception as exc:
                logger.error(
                    "Failed to poll Telegram, retrying in %ss: %s",
                    backoff,
                    _redact(config.telegram_bot_token, str(exc)),
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            for update in updates:
                offset = max(offset, update["update_id"] + 1)
                if not should_handle(update, config.allowed_user_ids):
                    continue
                pool.submit(
                    handle_update,
                    update,
                    config,
                    connector,
                    chat_locks,
                    config.telegram_bot_token,
                    store,
                )


if __name__ == "__main__":
    main()
