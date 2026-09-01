import logging
import multiprocessing
import queue

logger = logging.getLogger(__name__)


def _llm_call_target(connector, messages, tools, result_queue):
    # Runs inside the child process: calls the connector and puts the outcome on the queue.
    try:
        result_queue.put(("ok", connector.complete(messages, tools)))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def run_llm_call(connector, messages: list, tools, timeout: int):
    # Runs connector.complete in its own process, returning ("ok", AssistantMessage) or
    # ("error"/"timeout", detail). Turning those into user-facing text is the caller's job; this
    # function's only responsibility is making the call killable.
    # Drains the result queue *before* joining: for large replies the child can block writing to a
    # full pipe until the parent reads it, so joining first would time out and kill an already-
    # successful child.
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_llm_call_target, args=(connector, messages, tools, result_queue)
    )
    process.start()
    try:
        status, payload = result_queue.get(timeout=timeout)
    except queue.Empty:
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        return "timeout", f"no response within {timeout}s"
    process.join()
    if status == "ok":
        _log_usage(payload.usage)
    return status, payload


def _log_usage(usage) -> None:
    # Records the token cost of one call. Backends disagree on the key names (OpenAI-compatible
    # servers report prompt/completion_tokens, some report input/output_tokens) and a few omit
    # usage entirely, so missing counts are logged as "?" rather than guessed at.
    usage = usage or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    logger.info(
        "LLM call: input=%s output=%s",
        "?" if prompt is None else prompt,
        "?" if completion is None else completion,
    )
