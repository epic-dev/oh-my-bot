import logging
import time
from concurrent.futures import ThreadPoolExecutor

from config import load_config
from llm_client import OpenAICompatConnector
from telegram_client import get_updates
from worker import ChatLocks, handle_update

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Loads config, starts the thread pool, and long-polls Telegram forever, dispatching updates to workers.
    config = load_config()
    connector = OpenAICompatConnector(config.llm_base_url, config.llm_model)
    chat_locks = ChatLocks()
    offset = 0
    backoff = 1

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        while True:
            try:
                updates = get_updates(config.telegram_bot_token, offset, config.poll_timeout_seconds)
                backoff = 1
            except Exception:
                logger.exception("Failed to poll Telegram, retrying in %ss", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            for update in updates:
                offset = max(offset, update["update_id"] + 1)
                pool.submit(handle_update, update, config, connector, chat_locks, config.telegram_bot_token)


if __name__ == "__main__":
    main()
