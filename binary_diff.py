#!/usr/bin/env python3
"""Compare two unstripped Ghidra programs and export function/CFG differences.

This script is intended to run as a Ghidra/PyGhidra postScript.  It assumes
that both binaries retain useful symbols, so the primary function identity is
the fully-qualified function name.  Duplicate names are disambiguated by the
function signature when possible and are otherwise reported as ambiguous.

The script deliberately keeps three different notions of equality separate:

* layout hash: block offsets/sizes and typed edges;
* topology hash: an address-independent, typed Weisfeiler-Lehman fingerprint;
* instruction hash: topology annotated with normalized disassembly text.

Hashes are only fast summaries.  Block and edge CSV files contain the actual
mapping and differences used for the report.
"""

from __future__ import print_function

import hashlib
import os
import re
import sys
import time
import traceback

try:
    from ghidra.program.model.block import BasicBlockModel
    from ghidra.util.exception import CancelledException
    from ghidra.util.task import ConsoleTaskMonitor
    GHIDRA_AVAILABLE = True
except ImportError:
    # Keeping the pure comparison helpers importable makes them unit-testable
    # outside Ghidra.  main() still refuses to run without the Ghidra runtime.
    BasicBlockModel = None
    ConsoleTaskMonitor = None
    GHIDRA_AVAILABLE = False

    class CancelledException(Exception):
        pass


WL_MAX_ROUNDS = 8


# ghidra_common lives next to this script; when run as a Ghidra postScript the
# script directory is not always on sys.path.  The module imports nothing from
# ghidra.*, so it stays importable for unit tests outside Ghidra as well.
try:
    from ghidra_common import (
        address_text,
        atomic_write_csv,
        atomic_write_json,
        collect_block_instructions,
        flow_properties,
        get_block_start,
        get_script_arguments,
        java_iter,
        method_bool,
        safe_call,
    )
except ImportError:
    try:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _script_dir = os.getcwd()
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from ghidra_common import (
        address_text,
        atomic_write_csv,
        atomic_write_json,
        collect_block_instructions,
        flow_properties,
        get_block_start,
        get_script_arguments,
        java_iter,
        method_bool,
        safe_call,
    )


if GHIDRA_AVAILABLE:
    try:
        SCRIPT_MONITOR = monitor
    except NameError:
        SCRIPT_MONITOR = ConsoleTaskMonitor()
else:
    SCRIPT_MONITOR = None


AUTO_SYMBOL_RE = re.compile(
    r"\b(FUN|LAB|DAT|PTR|UNK|SUB|switchD|caseD|off)_[0-9A-Fa-f]+\b"
)


def log(message):
    print(message)


def check_cancelled():
    if SCRIPT_MONITOR is not None:
        SCRIPT_MONITOR.checkCancelled()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_parts(parts):
    return sha256_text("\x1f".join(str(part) for part in parts))


def relative_offset(address, entry):
    try:
        return int(address.subtract(entry))
    except Exception:
        try:
            return int(address.getOffset()) - int(entry.getOffset())
        except Exception:
            raise RuntimeError(
                "Cannot compute relative offset: {} - {}".format(address, entry)
            )


def signed_hex(value):
    value = int(value)
    if value < 0:
        return "-0x{:x}".format(-value)
    return "0x{:x}".format(value)


def normalize_instruction_text(instruction):
    """Normalize whitespace and address-derived Ghidra auto-symbols.

    Constants are intentionally retained: changing a comparison threshold or
    immediate value is a content change.  This is an instruction-text hash,
    not a proof of semantic equivalence.
    """
    text = str(instruction)
    text = AUTO_SYMBOL_RE.sub(lambda match: match.group(1) + "_<ADDR>", text)
    return " ".join(text.split())


def classify_edge(flow, source_instruction):
    properties = flow_properties(flow)
    source_properties = flow_properties(safe_call(source_instruction, "getFlowType"))

    if properties["fallthrough"]:
        if source_properties["call"]:
            return "call_fallthrough"
        if source_properties["conditional"]:
            return "conditional_not_taken"
        return "fallthrough"
    if properties["call"]:
        return "indirect_call" if properties["computed"] else "direct_call"
    if properties["jump"]:
        if properties["conditional"]:
            return "conditional_taken"
        return "indirect_jump" if properties["computed"] else "direct_jump"
    if properties["terminal"]:
        return "terminal"
    text = str(flow) if flow is not None else "unknown"
    return "unknown:" + text


def get_all_functions(program):
    functions = []
    iterator = program.getFunctionManager().getFunctions(True)
    for function in java_iter(iterator):
        check_cancelled()
        functions.append(function)
    return functions


def get_qualified_function_name(function):
    try:
        return str(function.getName(True))
    except Exception:
        symbol = safe_call(function, "getSymbol")
        try:
            return str(symbol.getName(True))
        except Exception:
            return str(safe_call(function, "getName", ""))


def get_function_signature(function):
    signature = safe_call(function, "getSignature")
    if signature is None:
        return ""
    prototype = safe_call(signature, "getPrototypeString")
    return str(prototype if prototype is not None else signature)


def describe_function(function):
    body = function.getBody()
    return {
        "object": function,
        "qualified_name": get_qualified_function_name(function),
        "signature": get_function_signature(function),
        "entry": address_text(safe_call(function, "getEntryPoint")),
        "body_size": int(safe_call(body, "getNumAddresses", 0) or 0),
        "parameter_count": int(safe_call(function, "getParameterCount", 0) or 0),
        "is_external": method_bool(function, "isExternal"),
        "is_thunk": method_bool(function, "isThunk"),
    }


def build_function_groups(descriptors):
    groups = {}
    for descriptor in descriptors:
        groups.setdefault(descriptor["qualified_name"], []).append(descriptor)
    return groups


def _unique_secondary_matches(old_items, new_items, key_function):
    old_groups = {}
    new_groups = {}
    for item in old_items:
        old_groups.setdefault(key_function(item), []).append(item)
    for item in new_items:
        new_groups.setdefault(key_function(item), []).append(item)

    pairs = []
    used_old = set()
    used_new = set()
    for key in sorted(set(old_groups) & set(new_groups), key=str):
        if not key:
            continue
        if len(old_groups[key]) == 1 and len(new_groups[key]) == 1:
            old_item = old_groups[key][0]
            new_item = new_groups[key][0]
            pairs.append((old_item, new_item))
            used_old.add(old_item["entry"])
            used_new.add(new_item["entry"])
    return (
        pairs,
        [item for item in old_items if item["entry"] not in used_old],
        [item for item in new_items if item["entry"] not in used_new],
    )


def match_functions(old_functions, new_functions):
    """Match unstripped functions without forcing ambiguous duplicate names."""
    old_descriptors = [describe_function(function) for function in old_functions]
    new_descriptors = [describe_function(function) for function in new_functions]
    old_groups = build_function_groups(old_descriptors)
    new_groups = build_function_groups(new_descriptors)

    matches = []
    old_only = []
    new_only = []
    ambiguous = []

    for qualified_name in sorted(set(old_groups) | set(new_groups)):
        old_items = list(old_groups.get(qualified_name, []))
        new_items = list(new_groups.get(qualified_name, []))

        if not old_items:
            new_only.extend(new_items)
            continue
        if not new_items:
            old_only.extend(old_items)
            continue

        if len(old_items) == 1 and len(new_items) == 1:
            matches.append({
                "qualified_name": qualified_name,
                "old": old_items[0],
                "new": new_items[0],
                "method": "qualified_name_unique",
                "confidence": "high",
            })
            continue

        signature_pairs, old_items, new_items = _unique_secondary_matches(
            old_items, new_items, lambda item: item["signature"]
        )
        for old_item, new_item in signature_pairs:
            matches.append({
                "qualified_name": qualified_name,
                "old": old_item,
                "new": new_item,
                "method": "qualified_name_and_signature",
                "confidence": "high",
            })

        # If signature matching leaves exactly one candidate on each side, the
        # common qualified-name group provides a deterministic residual match.
        if len(old_items) == 1 and len(new_items) == 1:
            matches.append({
                "qualified_name": qualified_name,
                "old": old_items[0],
                "new": new_items[0],
                "method": "qualified_name_residual",
                "confidence": "medium",
            })
            old_items = []
            new_items = []

        if old_items and new_items:
            ambiguous.append({
                "qualified_name": qualified_name,
                "old": old_items,
                "new": new_items,
                "reason": "duplicate qualified name could not be uniquely disambiguated",
            })
        else:
            old_only.extend(old_items)
            new_only.extend(new_items)

    matches.sort(key=lambda match: (
        match["qualified_name"], match["old"]["entry"], match["new"]["entry"]
    ))
    for index, match in enumerate(matches, 1):
        match["pair_id"] = "F{:06d}".format(index)

    return {
        "matches": matches,
        "old_only": old_only,
        "new_only": new_only,
        "ambiguous": ambiguous,
    }


def empty_cfg(descriptor, status="FAILED", phase="cfg", error=None):
    errors = []
    if error is not None:
        errors.append({
            "phase": phase,
            "function": descriptor["qualified_name"],
            "entry": descriptor["entry"],
            "block": "",
            "error_type": type(error).__name__,
            "message": str(error),
        })
    return {
        "function": descriptor,
        "status": status,
        "blocks": {},
        "edges": set(),
        "errors": errors,
        "layout_hash": "",
        "topology_hash": "",
        "mnemonic_hash": "",
        "instruction_hash": "",
    }


def calculate_degrees(blocks, edges):
    indegree = {block_id: 0 for block_id in blocks}
    outdegree = {block_id: 0 for block_id in blocks}
    for source, destination, _edge_type in edges:
        if source in outdegree:
            outdegree[source] += 1
        if destination in indegree:
            indegree[destination] += 1
    return indegree, outdegree


def graph_fingerprint(blocks, edges, annotation):
    """Return an address-independent typed WL graph fingerprint."""
    if not blocks:
        return sha256_text("EMPTY_GRAPH")

    incoming = {block_id: [] for block_id in blocks}
    outgoing = {block_id: [] for block_id in blocks}
    for source, destination, edge_type in edges:
        if source in outgoing and destination in incoming:
            outgoing[source].append((edge_type, destination))
            incoming[destination].append((edge_type, source))

    colors = {}
    for block_id, block in blocks.items():
        base = [
            "ENTRY" if block["relative_offset"] == 0 else "NODE",
            block["terminal_flow"],
            len(incoming[block_id]),
            len(outgoing[block_id]),
        ]
        if annotation == "mnemonic":
            base.append(block["mnemonic_hash"])
        elif annotation == "instruction":
            base.append(block["instruction_hash"])
        colors[block_id] = hash_parts(base)

    rounds = min(max(len(blocks), 1), WL_MAX_ROUNDS)
    for _round in range(rounds):
        refined = {}
        for block_id in blocks:
            in_labels = sorted(
                "{}:{}".format(edge_type, colors[source])
                for edge_type, source in incoming[block_id]
            )
            out_labels = sorted(
                "{}:{}".format(edge_type, colors[destination])
                for edge_type, destination in outgoing[block_id]
            )
            refined[block_id] = hash_parts(
                [colors[block_id], "IN"] + in_labels + ["OUT"] + out_labels
            )
        colors = refined

    node_summary = sorted(colors.values())
    edge_summary = sorted(
        "{}:{}:{}".format(colors[source], edge_type, colors[destination])
        for source, destination, edge_type in edges
        if source in colors and destination in colors
    )
    return hash_parts(
        ["NODES", len(blocks)] + node_summary + ["EDGES", len(edges)] + edge_summary
    )


def finalize_cfg_hashes(cfg):
    blocks = cfg["blocks"]
    edges = cfg["edges"]

    layout_parts = []
    for block in sorted(blocks.values(), key=lambda value: value["relative_offset"]):
        layout_parts.append(
            "B:{}:{}:{}".format(
                block["relative_offset"], block["address_count"], block["instruction_count"]
            )
        )
    for source, destination, edge_type in sorted(edges):
        layout_parts.append(
            "E:{}:{}:{}".format(
                blocks[source]["relative_offset"],
                blocks[destination]["relative_offset"],
                edge_type,
            )
        )

    cfg["layout_hash"] = hash_parts(layout_parts)
    cfg["topology_hash"] = graph_fingerprint(blocks, edges, "topology")
    cfg["mnemonic_hash"] = graph_fingerprint(blocks, edges, "mnemonic")
    cfg["instruction_hash"] = graph_fingerprint(blocks, edges, "instruction")


def extract_function_cfg(program, block_model, descriptor):
    function = descriptor["object"]
    entry = function.getEntryPoint()
    body = function.getBody()
    cfg = empty_cfg(descriptor, status="OK")
    block_objects = {}
    last_instructions = {}

    try:
        iterator = block_model.getCodeBlocksContaining(body, SCRIPT_MONITOR)
        for block in java_iter(iterator):
            check_cancelled()
            start = get_block_start(block)
            if start is None or not body.contains(start):
                continue

            block_id = address_text(start)
            instructions = collect_block_instructions(program.getListing(), block)
            mnemonic_parts = []
            instruction_parts = []
            for instruction in instructions:
                mnemonic_parts.append(
                    str(safe_call(instruction, "getMnemonicString", "")).upper()
                )
                instruction_parts.append(normalize_instruction_text(instruction))

            last_instruction = instructions[-1] if instructions else None
            terminal_flow = str(safe_call(last_instruction, "getFlowType", "EMPTY"))
            cfg["blocks"][block_id] = {
                "id": block_id,
                "relative_offset": relative_offset(start, entry),
                "address_count": int(safe_call(block, "getNumAddresses", 0) or 0),
                "instruction_count": len(instructions),
                "mnemonic_hash": hash_parts(mnemonic_parts),
                "instruction_hash": hash_parts(instruction_parts),
                "terminal_flow": terminal_flow,
            }
            block_objects[block_id] = block
            last_instructions[block_id] = last_instruction
    except CancelledException:
        raise
    except Exception as error:
        return empty_cfg(descriptor, status="FAILED", phase="collect_blocks", error=error)

    for source_id, block in block_objects.items():
        check_cancelled()
        try:
            destinations = block.getDestinations(SCRIPT_MONITOR)
            for reference in java_iter(destinations):
                destination = reference.getDestinationAddress()
                if destination is None or not body.contains(destination):
                    continue

                flow = reference.getFlowType()
                if flow_properties(flow)["call"]:
                    # A recursive/local call is not an intraprocedural CFG edge.
                    continue

                destination_block = safe_call(reference, "getDestinationBlock")
                if destination_block is None:
                    destination_block = block_model.getFirstCodeBlockContaining(
                        destination, SCRIPT_MONITOR
                    )
                destination_start = get_block_start(destination_block)
                destination_id = address_text(destination_start)
                if destination_id not in cfg["blocks"]:
                    cfg["errors"].append({
                        "phase": "resolve_destination_block",
                        "function": descriptor["qualified_name"],
                        "entry": descriptor["entry"],
                        "block": source_id,
                        "error_type": "MissingDestinationBlock",
                        "message": "Destination {} has no block in function body".format(
                            destination
                        ),
                    })
                    continue

                source_address = safe_call(reference, "getReferent")
                if source_address is None:
                    source_address = safe_call(reference, "getSourceAddress")
                source_instruction = None
                if source_address is not None:
                    source_instruction = program.getListing().getInstructionContaining(
                        source_address
                    )
                if source_instruction is None:
                    source_instruction = last_instructions.get(source_id)
                edge_type = classify_edge(flow, source_instruction)
                cfg["edges"].add((source_id, destination_id, edge_type))
        except CancelledException:
            raise
        except Exception as error:
            cfg["errors"].append({
                "phase": "collect_edges",
                "function": descriptor["qualified_name"],
                "entry": descriptor["entry"],
                "block": source_id,
                "error_type": type(error).__name__,
                "message": str(error),
            })

    if cfg["errors"]:
        cfg["status"] = "PARTIAL"
    finalize_cfg_hashes(cfg)
    return cfg


def extract_program_cfgs(program, functions, side):
    block_model = BasicBlockModel(program)
    cfgs = {}
    total = len(functions)
    for index, function in enumerate(functions, 1):
        check_cancelled()
        descriptor = describe_function(function)
        try:
            cfg = extract_function_cfg(program, block_model, descriptor)
        except CancelledException:
            raise
        except Exception as error:
            cfg = empty_cfg(descriptor, status="FAILED", phase="extract_cfg", error=error)
        cfgs[descriptor["entry"]] = cfg
        if index == 1 or index % 100 == 0 or index == total:
            log("[+] {} CFG progress: {}/{}".format(side, index, total))
    return cfgs


def block_degrees(cfg):
    return calculate_degrees(cfg["blocks"], cfg["edges"])


def _add_mapping(mapping, reverse_mapping, old_id, new_id, method):
    if old_id in mapping or new_id in reverse_mapping:
        return False
    mapping[old_id] = {"new_id": new_id, "method": method}
    reverse_mapping[new_id] = old_id
    return True


def _adjacency(cfg, direction):
    adjacency = {block_id: [] for block_id in cfg["blocks"]}
    for source, destination, edge_type in cfg["edges"]:
        if direction == "out":
            adjacency[source].append((edge_type, destination))
        else:
            adjacency[destination].append((edge_type, source))
    return adjacency


def _propagate_block_mapping(old_cfg, new_cfg, mapping, reverse_mapping):
    old_out = _adjacency(old_cfg, "out")
    new_out = _adjacency(new_cfg, "out")
    old_in = _adjacency(old_cfg, "in")
    new_in = _adjacency(new_cfg, "in")

    changed_any = False
    while True:
        changed = False
        for old_parent, mapping_record in list(mapping.items()):
            new_parent = mapping_record["new_id"]
            for old_adj, new_adj, method in (
                (old_out, new_out, "mapped_neighbor_out"),
                (old_in, new_in, "mapped_neighbor_in"),
            ):
                old_by_type = {}
                new_by_type = {}
                for edge_type, block_id in old_adj.get(old_parent, []):
                    if block_id not in mapping:
                        old_by_type.setdefault(edge_type, []).append(block_id)
                for edge_type, block_id in new_adj.get(new_parent, []):
                    if block_id not in reverse_mapping:
                        new_by_type.setdefault(edge_type, []).append(block_id)
                for edge_type in set(old_by_type) & set(new_by_type):
                    if len(old_by_type[edge_type]) == 1 and len(new_by_type[edge_type]) == 1:
                        changed |= _add_mapping(
                            mapping,
                            reverse_mapping,
                            old_by_type[edge_type][0],
                            new_by_type[edge_type][0],
                            method,
                        )
        changed_any |= changed
        if not changed:
            return changed_any


def block_similarity(old_block, new_block, old_degree, new_degree):
    score = 0.0
    if old_block["instruction_hash"] == new_block["instruction_hash"]:
        score += 0.30
    if old_block["mnemonic_hash"] == new_block["mnemonic_hash"]:
        score += 0.25
    if old_block["terminal_flow"] == new_block["terminal_flow"]:
        score += 0.15
    if old_degree == new_degree:
        score += 0.15
    else:
        difference = abs(old_degree[0] - new_degree[0]) + abs(old_degree[1] - new_degree[1])
        score += 0.15 / (1.0 + difference)

    old_count = old_block["instruction_count"]
    new_count = new_block["instruction_count"]
    maximum = max(old_count, new_count, 1)
    score += 0.10 * (1.0 - abs(old_count - new_count) / float(maximum))
    if old_block["relative_offset"] == new_block["relative_offset"]:
        score += 0.05
    return score


def _mutual_best_block_matches(old_cfg, new_cfg, mapping, reverse_mapping):
    old_indegree, old_outdegree = block_degrees(old_cfg)
    new_indegree, new_outdegree = block_degrees(new_cfg)
    old_ids = [block_id for block_id in old_cfg["blocks"] if block_id not in mapping]
    new_ids = [block_id for block_id in new_cfg["blocks"] if block_id not in reverse_mapping]

    scores = {}
    for old_id in old_ids:
        for new_id in new_ids:
            scores[(old_id, new_id)] = block_similarity(
                old_cfg["blocks"][old_id],
                new_cfg["blocks"][new_id],
                (old_indegree[old_id], old_outdegree[old_id]),
                (new_indegree[new_id], new_outdegree[new_id]),
            )

    def unique_best(candidates):
        ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
        if not ranked or ranked[0][1] < 0.60:
            return None
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.05:
            return None
        return ranked[0][0]

    best_new_for_old = {}
    for old_id in old_ids:
        best_new_for_old[old_id] = unique_best([
            (new_id, scores[(old_id, new_id)]) for new_id in new_ids
        ])
    best_old_for_new = {}
    for new_id in new_ids:
        best_old_for_new[new_id] = unique_best([
            (old_id, scores[(old_id, new_id)]) for old_id in old_ids
        ])

    changed = False
    for old_id, new_id in best_new_for_old.items():
        if new_id is not None and best_old_for_new.get(new_id) == old_id:
            changed |= _add_mapping(
                mapping, reverse_mapping, old_id, new_id, "mutual_best_similarity"
            )
    return changed


def map_basic_blocks(old_cfg, new_cfg):
    mapping = {}
    reverse_mapping = {}
    old_indegree, old_outdegree = block_degrees(old_cfg)
    new_indegree, new_outdegree = block_degrees(new_cfg)

    old_entries = [
        block_id for block_id, block in old_cfg["blocks"].items()
        if block["relative_offset"] == 0
    ]
    new_entries = [
        block_id for block_id, block in new_cfg["blocks"].items()
        if block["relative_offset"] == 0
    ]
    if len(old_entries) == 1 and len(new_entries) == 1:
        _add_mapping(mapping, reverse_mapping, old_entries[0], new_entries[0], "entry")

    def side_unique(field, method, include_offset=False):
        old_groups = {}
        new_groups = {}
        for block_id, block in old_cfg["blocks"].items():
            if block_id in mapping:
                continue
            key = (
                block[field], block["terminal_flow"], old_indegree[block_id], old_outdegree[block_id]
            )
            if include_offset:
                key += (block["relative_offset"],)
            old_groups.setdefault(key, []).append(block_id)
        for block_id, block in new_cfg["blocks"].items():
            if block_id in reverse_mapping:
                continue
            key = (
                block[field], block["terminal_flow"], new_indegree[block_id], new_outdegree[block_id]
            )
            if include_offset:
                key += (block["relative_offset"],)
            new_groups.setdefault(key, []).append(block_id)
        for key in set(old_groups) & set(new_groups):
            if len(old_groups[key]) == 1 and len(new_groups[key]) == 1:
                _add_mapping(
                    mapping, reverse_mapping, old_groups[key][0], new_groups[key][0], method
                )

    side_unique("instruction_hash", "unique_instruction_block")
    side_unique("mnemonic_hash", "unique_mnemonic_block")
    _propagate_block_mapping(old_cfg, new_cfg, mapping, reverse_mapping)
    side_unique("mnemonic_hash", "same_relative_structure", include_offset=True)

    while _mutual_best_block_matches(old_cfg, new_cfg, mapping, reverse_mapping):
        _propagate_block_mapping(old_cfg, new_cfg, mapping, reverse_mapping)

    return mapping, reverse_mapping


def compare_cfg_pair(match, old_cfg, new_cfg):
    mapping, reverse_mapping = map_basic_blocks(old_cfg, new_cfg)
    old_blocks = old_cfg["blocks"]
    new_blocks = new_cfg["blocks"]
    block_rows = []
    counters = {
        "blocks_added": 0,
        "blocks_removed": 0,
        "blocks_modified": 0,
        "blocks_moved": 0,
        "edges_added": 0,
        "edges_removed": 0,
        "edges_type_changed": 0,
    }

    for old_id in sorted(old_blocks, key=lambda value: old_blocks[value]["relative_offset"]):
        old_block = old_blocks[old_id]
        mapping_record = mapping.get(old_id)
        if mapping_record is None:
            counters["blocks_removed"] += 1
            block_rows.append([
                match["pair_id"], match["qualified_name"], "REMOVED", "",
                old_id, signed_hex(old_block["relative_offset"]), "", "",
                old_block["instruction_count"], "", old_block["address_count"], "",
                old_block["mnemonic_hash"], "", old_block["instruction_hash"], "",
            ])
            continue

        new_id = mapping_record["new_id"]
        new_block = new_blocks[new_id]
        content_equal = old_block["instruction_hash"] == new_block["instruction_hash"]
        moved = old_block["relative_offset"] != new_block["relative_offset"]
        if content_equal and not moved:
            status = "UNCHANGED"
        elif content_equal:
            status = "MOVED"
            counters["blocks_moved"] += 1
        elif moved:
            status = "MODIFIED_MOVED"
            counters["blocks_modified"] += 1
            counters["blocks_moved"] += 1
        else:
            status = "MODIFIED"
            counters["blocks_modified"] += 1

        block_rows.append([
            match["pair_id"], match["qualified_name"], status, mapping_record["method"],
            old_id, signed_hex(old_block["relative_offset"]),
            new_id, signed_hex(new_block["relative_offset"]),
            old_block["instruction_count"], new_block["instruction_count"],
            old_block["address_count"], new_block["address_count"],
            old_block["mnemonic_hash"], new_block["mnemonic_hash"],
            old_block["instruction_hash"], new_block["instruction_hash"],
        ])

    for new_id in sorted(new_blocks, key=lambda value: new_blocks[value]["relative_offset"]):
        if new_id in reverse_mapping:
            continue
        new_block = new_blocks[new_id]
        counters["blocks_added"] += 1
        block_rows.append([
            match["pair_id"], match["qualified_name"], "ADDED", "",
            "", "", new_id, signed_hex(new_block["relative_offset"]),
            "", new_block["instruction_count"], "", new_block["address_count"],
            "", new_block["mnemonic_hash"], "", new_block["instruction_hash"],
        ])

    edge_rows = []
    remaining_new_edges = set(new_cfg["edges"])
    deferred_removed = []
    for old_edge in sorted(old_cfg["edges"]):
        old_source, old_destination, old_type = old_edge
        if old_source in mapping and old_destination in mapping:
            new_source = mapping[old_source]["new_id"]
            new_destination = mapping[old_destination]["new_id"]
            expected = (new_source, new_destination, old_type)
            if expected in remaining_new_edges:
                remaining_new_edges.remove(expected)
                continue
            candidates = sorted(
                edge for edge in remaining_new_edges
                if edge[0] == new_source and edge[1] == new_destination
            )
            if len(candidates) == 1:
                new_edge = candidates[0]
                remaining_new_edges.remove(new_edge)
                counters["edges_type_changed"] += 1
                edge_rows.append([
                    match["pair_id"], match["qualified_name"], "TYPE_CHANGED",
                    old_source, old_destination, old_type,
                    new_edge[0], new_edge[1], new_edge[2],
                ])
                continue
        deferred_removed.append(old_edge)

    for old_source, old_destination, old_type in deferred_removed:
        counters["edges_removed"] += 1
        edge_rows.append([
            match["pair_id"], match["qualified_name"], "REMOVED",
            old_source, old_destination, old_type, "", "", "",
        ])
    for new_source, new_destination, new_type in sorted(remaining_new_edges):
        counters["edges_added"] += 1
        edge_rows.append([
            match["pair_id"], match["qualified_name"], "ADDED",
            "", "", "", new_source, new_destination, new_type,
        ])

    mapping_complete = (
        len(mapping) == len(old_blocks) and len(reverse_mapping) == len(new_blocks)
    )
    topology_equal = old_cfg["topology_hash"] == new_cfg["topology_hash"]
    layout_equal = old_cfg["layout_hash"] == new_cfg["layout_hash"]
    instruction_equal = old_cfg["instruction_hash"] == new_cfg["instruction_hash"]

    if old_cfg["status"] == "FAILED" or new_cfg["status"] == "FAILED":
        status = "ANALYSIS_FAILED"
    elif old_cfg["status"] == "PARTIAL" or new_cfg["status"] == "PARTIAL":
        status = "ANALYSIS_PARTIAL"
    elif not mapping_complete and topology_equal:
        status = "BLOCK_MAPPING_AMBIGUOUS"
    else:
        structural_change = any(counters[key] for key in (
            "blocks_added", "blocks_removed", "edges_added",
            "edges_removed", "edges_type_changed"
        ))
        if structural_change or not topology_equal:
            status = "CFG_CHANGED"
        elif counters["blocks_modified"] or not instruction_equal:
            status = "CONTENT_ONLY"
        elif counters["blocks_moved"] or not layout_equal:
            status = "LAYOUT_ONLY"
        else:
            status = "UNCHANGED"

    summary = {
        "pair_id": match["pair_id"],
        "qualified_name": match["qualified_name"],
        "old_entry": match["old"]["entry"],
        "new_entry": match["new"]["entry"],
        "match_method": match["method"],
        "match_confidence": match["confidence"],
        "status": status,
        "old_analysis_status": old_cfg["status"],
        "new_analysis_status": new_cfg["status"],
        "old_block_count": len(old_blocks),
        "new_block_count": len(new_blocks),
        "old_edge_count": len(old_cfg["edges"]),
        "new_edge_count": len(new_cfg["edges"]),
        "mapped_block_count": len(mapping),
        "mapping_complete": mapping_complete,
        "topology_equal": topology_equal,
        "layout_equal": layout_equal,
        "instruction_equal": instruction_equal,
        "old_layout_hash": old_cfg["layout_hash"],
        "new_layout_hash": new_cfg["layout_hash"],
        "old_topology_hash": old_cfg["topology_hash"],
        "new_topology_hash": new_cfg["topology_hash"],
        "old_mnemonic_hash": old_cfg["mnemonic_hash"],
        "new_mnemonic_hash": new_cfg["mnemonic_hash"],
        "old_instruction_hash": old_cfg["instruction_hash"],
        "new_instruction_hash": new_cfg["instruction_hash"],
    }
    summary.update(counters)
    return summary, block_rows, edge_rows


def function_match_rows(match_result):
    rows = []
    for match in match_result["matches"]:
        rows.append([
            "MATCHED", match["pair_id"], match["qualified_name"],
            match["old"]["entry"], match["new"]["entry"],
            match["old"]["signature"], match["new"]["signature"],
            match["method"], match["confidence"], "",
        ])
    for descriptor in match_result["old_only"]:
        rows.append([
            "OLD_ONLY", "", descriptor["qualified_name"], descriptor["entry"], "",
            descriptor["signature"], "", "", "", "",
        ])
    for descriptor in match_result["new_only"]:
        rows.append([
            "NEW_ONLY", "", descriptor["qualified_name"], "", descriptor["entry"],
            "", descriptor["signature"], "", "", "",
        ])
    for group in match_result["ambiguous"]:
        rows.append([
            "AMBIGUOUS", "", group["qualified_name"],
            ";".join(item["entry"] for item in group["old"]),
            ";".join(item["entry"] for item in group["new"]),
            ";".join(item["signature"] for item in group["old"]),
            ";".join(item["signature"] for item in group["new"]),
            "", "", group["reason"],
        ])
    return rows


def unmatched_function_rows(match_result):
    rows = []
    for side, descriptors in (
        ("OLD_ONLY", match_result["old_only"]),
        ("NEW_ONLY", match_result["new_only"]),
    ):
        for descriptor in descriptors:
            rows.append([
                side, descriptor["qualified_name"], descriptor["entry"],
                descriptor["body_size"], descriptor["parameter_count"],
                int(descriptor["is_external"]), int(descriptor["is_thunk"]),
            ])
    for group in match_result["ambiguous"]:
        for side, descriptors in (("AMBIGUOUS_OLD", group["old"]), ("AMBIGUOUS_NEW", group["new"])):
            for descriptor in descriptors:
                rows.append([
                    side, descriptor["qualified_name"], descriptor["entry"],
                    descriptor["body_size"], descriptor["parameter_count"],
                    int(descriptor["is_external"]), int(descriptor["is_thunk"]),
                ])
    return rows


def cfg_hash_rows(cfgs):
    rows = []
    for entry, cfg in sorted(cfgs.items()):
        descriptor = cfg["function"]
        rows.append([
            descriptor["qualified_name"], entry, cfg["status"],
            len(cfg["blocks"]), len(cfg["edges"]),
            cfg["layout_hash"], cfg["topology_hash"],
            cfg["mnemonic_hash"], cfg["instruction_hash"], len(cfg["errors"]),
        ])
    return rows


def analysis_error_rows(old_cfgs, new_cfgs):
    rows = []
    for side, cfgs in (("OLD", old_cfgs), ("NEW", new_cfgs)):
        for cfg in cfgs.values():
            for error in cfg["errors"]:
                rows.append([
                    side, error["phase"], error["function"], error["entry"],
                    error["block"], error["error_type"], error["message"],
                ])
    return rows


def program_metadata(program):
    language = safe_call(program, "getLanguage")
    compiler = safe_call(program, "getCompilerSpec")
    return {
        "name": str(safe_call(program, "getName", "")),
        "executable_path": str(safe_call(program, "getExecutablePath", "")),
        "sha256": str(safe_call(program, "getExecutableSHA256", "") or ""),
        "md5": str(safe_call(program, "getExecutableMD5", "") or ""),
        "image_base": address_text(safe_call(program, "getImageBase")),
        "language_id": str(safe_call(language, "getLanguageID", "")),
        "compiler_spec": str(safe_call(compiler, "getCompilerSpecID", "")),
    }


def validate_program_compatibility(old_program, new_program):
    old_metadata = program_metadata(old_program)
    new_metadata = program_metadata(new_program)
    if old_metadata["language_id"] != new_metadata["language_id"]:
        raise RuntimeError(
            "Program language mismatch: {} vs {}".format(
                old_metadata["language_id"], new_metadata["language_id"]
            )
        )
    if old_metadata["compiler_spec"] != new_metadata["compiler_spec"]:
        log("[!] Compiler spec differs: {} vs {}".format(
            old_metadata["compiler_spec"], new_metadata["compiler_spec"]
        ))
    return old_metadata, new_metadata


def normalize_project_path(value):
    return str(value).replace("\\", "/").strip("/")


def get_project_root_folder_from_current_program():
    folder = currentProgram.getDomainFile().getParent()
    while True:
        parent = folder.getParent()
        if parent is None:
            return folder
        folder = parent


def find_domain_files(folder, target_name):
    target = normalize_project_path(target_name)
    has_path = "/" in target
    matches = []

    for domain_file in folder.getFiles():
        pathname = normalize_project_path(domain_file.getPathname())
        if has_path:
            if pathname == target:
                matches.append(domain_file)
        elif domain_file.getName() == target:
            matches.append(domain_file)

    for subfolder in folder.getFolders():
        matches.extend(find_domain_files(subfolder, target))
    return matches


def resolve_domain_file(program_name):
    matches = find_domain_files(get_project_root_folder_from_current_program(), program_name)
    if not matches:
        raise RuntimeError("Cannot find program in project: {}".format(program_name))
    if len(matches) > 1:
        paths = ", ".join(sorted(str(domain_file.getPathname()) for domain_file in matches))
        raise RuntimeError(
            "Program name is ambiguous; use a project path: {} -> {}".format(
                program_name, paths
            )
        )
    return matches[0]


def resolve_program(program_name, cache):
    domain_file = resolve_domain_file(program_name)
    pathname = normalize_project_path(domain_file.getPathname())
    if pathname in cache:
        return cache[pathname]

    current_path = normalize_project_path(currentProgram.getDomainFile().getPathname())
    if pathname == current_path:
        result = (currentProgram, False)
    else:
        program = domain_file.getDomainObject(currentProgram, False, False, SCRIPT_MONITOR)
        if program is None or not hasattr(program, "getFunctionManager"):
            raise RuntimeError("Domain object is not a Program: {}".format(program_name))
        result = (program, True)
    cache[pathname] = result
    return result


def write_outputs(
    out_dir,
    match_result,
    old_cfgs,
    new_cfgs,
    function_summaries,
    block_rows,
    edge_rows,
    metadata,
):
    atomic_write_csv(
        os.path.join(out_dir, "function_matches.csv"),
        [
            "status", "pair_id", "qualified_name", "old_entry", "new_entry",
            "old_signature", "new_signature", "match_method", "confidence", "reason",
        ],
        function_match_rows(match_result),
    )
    atomic_write_csv(
        os.path.join(out_dir, "unmatched_functions.csv"),
        [
            "side", "qualified_name", "entry", "body_size", "parameter_count",
            "is_external", "is_thunk",
        ],
        unmatched_function_rows(match_result),
    )
    hash_header = [
        "qualified_name", "entry", "analysis_status", "basic_block_count", "edge_count",
        "layout_hash", "topology_hash", "mnemonic_hash", "instruction_hash", "error_count",
    ]
    atomic_write_csv(
        os.path.join(out_dir, "old_cfg_hash.csv"), hash_header, cfg_hash_rows(old_cfgs)
    )
    atomic_write_csv(
        os.path.join(out_dir, "new_cfg_hash.csv"), hash_header, cfg_hash_rows(new_cfgs)
    )

    function_header = [
        "pair_id", "qualified_name", "old_entry", "new_entry", "match_method",
        "match_confidence", "status", "old_analysis_status", "new_analysis_status",
        "old_block_count", "new_block_count", "old_edge_count", "new_edge_count",
        "mapped_block_count", "mapping_complete", "blocks_added", "blocks_removed",
        "blocks_modified", "blocks_moved", "edges_added", "edges_removed",
        "edges_type_changed", "topology_equal", "layout_equal", "instruction_equal",
        "old_layout_hash", "new_layout_hash", "old_topology_hash", "new_topology_hash",
        "old_mnemonic_hash", "new_mnemonic_hash", "old_instruction_hash",
        "new_instruction_hash",
    ]
    atomic_write_csv(
        os.path.join(out_dir, "cfg_function_diff.csv"),
        function_header,
        [[summary[column] for column in function_header] for summary in function_summaries],
    )
    atomic_write_csv(
        os.path.join(out_dir, "cfg_block_diff.csv"),
        [
            "pair_id", "qualified_name", "status", "mapping_method",
            "old_block", "old_relative_offset", "new_block", "new_relative_offset",
            "old_instruction_count", "new_instruction_count", "old_address_count",
            "new_address_count", "old_mnemonic_hash", "new_mnemonic_hash",
            "old_instruction_hash", "new_instruction_hash",
        ],
        block_rows,
    )
    atomic_write_csv(
        os.path.join(out_dir, "cfg_edge_diff.csv"),
        [
            "pair_id", "qualified_name", "status", "old_source", "old_destination",
            "old_edge_type", "new_source", "new_destination", "new_edge_type",
        ],
        edge_rows,
    )
    atomic_write_csv(
        os.path.join(out_dir, "analysis_errors.csv"),
        [
            "side", "phase", "function", "entry", "block", "error_type", "message",
        ],
        analysis_error_rows(old_cfgs, new_cfgs),
    )
    atomic_write_json(os.path.join(out_dir, "metadata.json"), metadata)


def print_usage():
    print("Usage 1:")
    print("  binary_diff.py <new_program_project_name> <out_dir>")
    print("")
    print("Usage 2:")
    print("  binary_diff.py <old_program_project_name> <new_program_project_name> <out_dir>")
    print("")
    print("Recommended with pyghidraRun:")
    print("  -process old.exe -postScript binary_diff.py new.exe /path/to/output")


def main():
    if not GHIDRA_AVAILABLE:
        raise RuntimeError("Ghidra runtime is unavailable")
    try:
        active_program = currentProgram
    except NameError:
        active_program = None
    if active_program is None:
        raise RuntimeError("currentProgram is unavailable; run as a Ghidra/PyGhidra script")

    arguments = get_script_arguments()
    if len(arguments) not in (2, 3):
        print_usage()
        return 2

    program_cache = {
        normalize_project_path(active_program.getDomainFile().getPathname()):
            (active_program, False)
    }
    opened_programs = []

    try:
        if len(arguments) == 2:
            old_program = active_program
            new_name, out_dir = arguments
        else:
            old_name, new_name, out_dir = arguments
            old_program, owns_old = resolve_program(old_name, program_cache)
            if owns_old:
                opened_programs.append(old_program)

        new_program, owns_new = resolve_program(new_name, program_cache)
        if owns_new and new_program not in opened_programs:
            opened_programs.append(new_program)

        ensure_dir(out_dir)
        old_metadata, new_metadata = validate_program_compatibility(old_program, new_program)

        log("[+] OLD program: {}".format(old_program.getName()))
        log("[+] NEW program: {}".format(new_program.getName()))
        log("[+] OUT dir: {}".format(out_dir))

        old_functions = get_all_functions(old_program)
        new_functions = get_all_functions(new_program)
        log("[+] OLD functions: {}".format(len(old_functions)))
        log("[+] NEW functions: {}".format(len(new_functions)))

        old_cfgs = extract_program_cfgs(old_program, old_functions, "OLD")
        new_cfgs = extract_program_cfgs(new_program, new_functions, "NEW")

        match_result = match_functions(old_functions, new_functions)
        matches = match_result["matches"]
        log("[+] Matched functions: {}".format(len(matches)))
        log("[+] Ambiguous function groups: {}".format(len(match_result["ambiguous"])))

        function_summaries = []
        block_rows = []
        edge_rows = []
        for match in matches:
            check_cancelled()
            summary, pair_block_rows, pair_edge_rows = compare_cfg_pair(
                match,
                old_cfgs[match["old"]["entry"]],
                new_cfgs[match["new"]["entry"]],
            )
            function_summaries.append(summary)
            block_rows.extend(pair_block_rows)
            edge_rows.extend(pair_edge_rows)

        status_counts = {}
        for summary in function_summaries:
            status_counts[summary["status"]] = status_counts.get(summary["status"], 0) + 1
        metadata = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "assumption": "unstripped binaries with stable qualified symbols",
            "old_program": old_metadata,
            "new_program": new_metadata,
            "function_counts": {
                "old": len(old_functions),
                "new": len(new_functions),
                "matched": len(matches),
                "old_only": len(match_result["old_only"]),
                "new_only": len(match_result["new_only"]),
                "ambiguous_groups": len(match_result["ambiguous"]),
            },
            "diff_status_counts": status_counts,
        }

        write_outputs(
            out_dir,
            match_result,
            old_cfgs,
            new_cfgs,
            function_summaries,
            block_rows,
            edge_rows,
            metadata,
        )

        log("[+] CFG diff status counts: {}".format(status_counts))
        log("[+] Done.")
        return 0
    finally:
        released = set()
        for program in reversed(opened_programs):
            pathname = normalize_project_path(program.getDomainFile().getPathname())
            if pathname in released:
                continue
            released.add(pathname)
            try:
                program.release(currentProgram)
            except Exception as error:
                log("[!] Failed to release program {}: {}".format(pathname, error))


if __name__ == "__main__":
    try:
        exit_code = main()
    except CancelledException:
        print("[!] Binary diff cancelled")
        raise SystemExit(130)
    except Exception as error:
        print("[!] Binary diff failed: {}".format(error))
        traceback.print_exc()
        raise
    if exit_code:
        raise SystemExit(exit_code)
