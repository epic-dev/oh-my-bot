import logging
import time

from .session import handle_command
from .telegram_client import send_message
from .tools import TOOL_SCHEMAS, ToolContext, dispatch
from .worker import run_llm_call

logger = logging.getLogger(__name__)

PROGRESS_RESULT_CHARS = 500
MAX_BACKOFF_SECONDS = 10


def run_turn(text, session, connector, config, approvals, token) -> None:
    # Drives one user message to a final answer: model call, tool calls, repeat, under three
    # independent circuit breakers. Sends every message to the chat itself.
    command_reply = handle_command(text, session)
    if command_reply is not None:
        send_message(token, session.chat_id, command_reply)
        return

    session.add_user(text)
    ctx = ToolContext(session=session, config=config, approvals=approvals)
    iterations = 0
    llm_failures = 0
    tool_failures = 0

    while True:
        if iterations >= config.max_loop_iterations:
            send_message(
                token,
                session.chat_id,
                f"I hit my step limit after {iterations} steps. "
                "Send another message to continue from here.",
            )
            return

        status, payload = run_llm_call(
            connector, session.history(), TOOL_SCHEMAS, config.llm_timeout_seconds
        )
        if status != "ok":
            llm_failures += 1
            logger.error(
                "LLM call failed (%s/%s): %s", llm_failures, config.max_llm_retries, payload
            )
            if llm_failures >= config.max_llm_retries:
                send_message(
                    token, session.chat_id, "I couldn't reach the AI service. Please try again shortly."
                )
                return
            # A transport failure is not a reasoning step, so it does not burn a loop iteration.
            time.sleep(min(2 ** llm_failures, MAX_BACKOFF_SECONDS))
            continue
        llm_failures = 0
        message = payload

        if not message.tool_calls:
            reply = message.content
            if not reply and message.finish_reason == "length":
                reply = (
                    "I ran out of room while thinking and never got to an answer. "
                    "Try asking more specifically, or raise LLM_MAX_TOKENS."
                )
            session.add_assistant(reply)
            send_message(token, session.chat_id, reply)
            return

        # The assistant turn carrying tool_calls is persisted BEFORE any tool runs, so the call
        # and its results stay paired in the history. If the process dies mid-tool the history
        # holds a call with no result, which is why a restart abandons in-flight turns.
        session.add_assistant(message.content, tool_calls=[c.to_wire() for c in message.tool_calls])
        for tool_call in message.tool_calls:
            output, ok = dispatch(tool_call, ctx)
            _send_progress(token, session.chat_id, tool_call, output)
            session.add_tool_result(tool_call.id, output)
            if ok:
                tool_failures = 0
                continue
            tool_failures += 1
            if tool_failures >= config.max_consecutive_tool_failures:
                send_message(
                    token,
                    session.chat_id,
                    f"A tool kept failing ({tool_failures} times in a row); stopping here.",
                )
                return
        iterations += 1


def _send_progress(token, chat_id, tool_call, output) -> None:
    # Posts one progress message per tool call: what was run, and a trimmed view of what came back.
    if tool_call.name == "exec":
        header = f"$ {tool_call.arguments.get('command', '')}"
    else:
        arguments = ", ".join(f"{k}=..." for k in tool_call.arguments)
        header = f"{tool_call.name}({arguments})"
    body = output if len(output) <= PROGRESS_RESULT_CHARS else output[:PROGRESS_RESULT_CHARS] + "\n[...]"
    send_message(token, chat_id, f"{header}\n\n{body}")
