import logging
import time

from .actors import ActorPool
from .agent import run_turn
from .approvals import ApprovalRegistry
from .config import load_config
from .llm_client import OpenAICompatConnector
from .session import Session
from .skills import load_skills
from .store import Store
from .telegram_client import _redact, get_updates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_allowed_message(update, allowed_user_ids) -> bool:
    # Decides whether one message update may be processed: it must be a text message, from an
    # allowlisted user, in a private chat. The allowlist keys on the *user* id, never the chat id —
    # in a group those differ, and a chat-id check would authorize every member of an allowed group.
    message = update.get("message")
    if not message or "text" not in message:
        return False
    user_id = (message.get("from") or {}).get("id")
    chat_type = (message.get("chat") or {}).get("type")
    if user_id not in allowed_user_ids or chat_type != "private":
        logger.info("Dropping message from user %s (chat type %s)", user_id, chat_type)
        return False
    return True


def _is_allowed_callback(update, allowed_user_ids) -> bool:
    # Decides whether one callback_query may be acted on. This check is as important as the one
    # for messages: without it anyone could approve another user's pending command by guessing a
    # request id.
    query = update.get("callback_query") or {}
    user_id = (query.get("from") or {}).get("id")
    if user_id not in allowed_user_ids:
        logger.info("Dropping callback query from user %s", user_id)
        return False
    return True


def route_update(update, config, approvals, actors) -> str:
    # Sends one update where it belongs and names what it did, for logging and tests:
    # button taps resolve a pending approval, text messages queue on their chat's actor.
    if "callback_query" in update:
        if not _is_allowed_callback(update, config.allowed_user_ids):
            return "dropped"
        return "callback" if approvals.resolve(update) else "stale-callback"
    if _is_allowed_message(update, config.allowed_user_ids):
        message = update["message"]
        actors.submit(message["chat"]["id"], (message["chat"]["id"], message["text"]))
        return "message"
    return "dropped"


def main():
    # Loads config, wires the store, connector, approvals and per-chat actors, then long-polls
    # Telegram forever, routing each update.
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
    approvals = ApprovalRegistry(config.telegram_bot_token, config.approval_timeout_seconds)
    skills = load_skills(config.skills_dir)

    def handle(item):
        # Runs one queued message on its chat's actor thread.
        chat_id, text = item
        session = Session(chat_id, store, config, skills)
        run_turn(text, session, connector, config, approvals, config.telegram_bot_token)

    actors = ActorPool(handle)
    offset = 0
    backoff = 1

    logger.info("Listening. Allowed users: %s", sorted(config.allowed_user_ids))
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
            try:
                route_update(update, config, approvals, actors)
            except Exception:
                logger.exception("Failed to route update %s", update.get("update_id"))


if __name__ == "__main__":
    main()
