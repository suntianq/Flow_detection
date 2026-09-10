"""Shared low-level helpers for the Ghidra/PyGhidra analysis scripts.

Only genuinely generic, business-agnostic utilities live here so that
extract_cfg.py and binary_diff.py can share one implementation.  Functions
whose semantics differ between the two scripts (logging, program metadata,
edge classification, ...) intentionally stay in their own modules.

This module must be importable both inside Ghidra (as a postScript sibling)
and as a plain module for unit testing outside Ghidra, so it imports nothing
from the ghidra.* packages at module load time.
"""

from __future__ import print_function

import csv
import json
import os
import sys


def safe_call(obj, method_name, default=None):
    """Call a zero-argument Java/Python method, returning default on failure.

    Guards against a None receiver as well as any exception raised by the
    call itself.
    """
    if obj is None:
        return default
    try:
        return getattr(obj, method_name)()
    except Exception:
        return default


def method_bool(obj, method_name):
    """Call a zero-argument predicate method and coerce the result to bool."""
    try:
        return bool(getattr(obj, method_name)())
    except Exception:
        return False


def java_iter(iterator):
    """Iterate either a Java iterator (hasNext/next) or a Python iterable."""
    if iterator is None:
        return
    try:
        while iterator.hasNext():
            yield iterator.next()
        return
    except AttributeError:
        pass
    for value in iterator:
        yield value


def bool_int(value):
    """Return 1 for truthy values and 0 otherwise (for CSV columns)."""
    return 1 if value else 0


def flow_properties(flow):
    """Summarize a Ghidra FlowType into a plain dict of boolean properties."""
    return {
        "call": method_bool(flow, "isCall"),
        "jump": method_bool(flow, "isJump"),
        "conditional": method_bool(flow, "isConditional"),
        "computed": method_bool(flow, "isComputed"),
        "terminal": method_bool(flow, "isTerminal"),
        "fallthrough": method_bool(flow, "isFallthrough"),
    }


def address_text(address):
    """Render an address as text, mapping None to the empty string."""
    return "" if address is None else str(address)


def get_block_start(block):
    """Return a code block's canonical start address, or None."""
    if block is None:
        return None
    start = safe_call(block, "getFirstStartAddress")
    if start is not None:
        return start
    return safe_call(block, "getMinAddress")


def collect_block_instructions(listing, block):
    """Collect the instructions of one code block in address order.

    Takes a Ghidra Listing directly; callers holding only a Program should
    pass ``program.getListing()``.
    """
    instructions = []
    iterator = listing.getInstructions(block, True)
    for instruction in java_iter(iterator):
        instructions.append(instruction)
    return instructions


def get_script_arguments():
    """Return Ghidra postScript arguments, falling back to sys.argv."""
    try:
        return list(getScriptArgs())  # noqa: F821 - injected by Ghidra
    except Exception:
        return sys.argv[1:]


def atomic_write_csv(path, header, rows, sort=False):
    """Write a CSV file atomically via a temporary file and os.replace.

    When ``sort`` is true the rows are written in a deterministic order
    (each row compared as a tuple of stringified cells); callers that have
    already ordered their rows leave it false to preserve their own order.
    """
    if sort:
        rows = sorted(rows, key=lambda value: tuple(str(item) for item in value))
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)
        csv_file.flush()
        try:
            os.fsync(csv_file.fileno())
        except Exception:
            pass
    os.replace(temporary_path, path)


def atomic_write_json(path, value):
    """Write a JSON file atomically via a temporary file and os.replace."""
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as json_file:
        json.dump(value, json_file, ensure_ascii=False, indent=2, sort_keys=True)
        json_file.write("\n")
        json_file.flush()
        try:
            os.fsync(json_file.fileno())
        except Exception:
            pass
    os.replace(temporary_path, path)
