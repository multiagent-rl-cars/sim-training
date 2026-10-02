"""Failure recovery that preserves the original training error."""

from collections.abc import Callable, Sequence


def best_effort_cleanup(
    original_error: BaseException,
    operations: Sequence[tuple[str, Callable[[], object]]],
) -> None:
    """Attempt every recovery operation and attach failures to the original error.

    The caller must re-raise the original exception. This function is for a
    failure path only: normal execution must not silently suppress errors.
    Exception class names, not potentially sensitive SDK error messages, are
    included in the notes.
    """
    for name, operation in operations:
        try:
            operation()
        except BaseException as cleanup_error:
            original_error.add_note(f"Recovery {name!r} failed: {type(cleanup_error).__name__}")


def finish_resources(
    operations: Sequence[tuple[str, Callable[[], object]]],
    original_error: BaseException | None = None,
) -> None:
    """Close all resources, preserving either the active or first cleanup error."""
    if original_error is not None:
        best_effort_cleanup(original_error, operations)
        return
    for index, (name, operation) in enumerate(operations):
        try:
            operation()
        except BaseException as error:
            error.add_note(f"Resource cleanup {name!r} failed")
            best_effort_cleanup(error, operations[index + 1 :])
            raise
