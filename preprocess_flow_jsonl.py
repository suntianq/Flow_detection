#!/usr/bin/env python3
"""Preprocess symbolized TRBE/ETM JSONL into structured flow sequences.

The symbolizer emits low-level events such as address, instruction_range,
context and flow_gap.  This preprocessor keeps only continuous target-ELF
control flow and converts each valid instruction_range into one edge token:

    branch/return/call instruction exit  ->  next_pc target entry

Each emitted segment stores the first basic-block entry once as segment.start.
Every token then stores only the control instruction location, the control
metadata and the destination entry.

For each input JSONL, the output directory receives one JSONL file with the
same file name.  Directory inputs are processed in batch, optionally using
multiple worker processes.  Summary statistics are printed to stdout; no stats
or debug JSONL files are written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, TextIO, Tuple


UNKNOWN = "<unknown>"
NORMAL_RANGE_STATUSES = {None, "", "resolved", "target_resolved"}
CUT_ADDRESS_REASONS = {
    "no-configured-elf",
    "address-not-in-maps",
    "address-in-non-executable-map",
}
SYMBOL_RE = re.compile(r"^(?P<name>.*?)(?P<relation>~?\+)(?P<offset>0x[0-9a-fA-F]+|\d+)$")


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text, 0)
        except ValueError:
            return None
    return None


def hex_or_none(value: Optional[int], *, width: int = 0) -> Optional[str]:
    if value is None:
        return None
    if width > 0:
        return f"0x{value:0{width}x}"
    return f"0x{value:x}"


def write_json(stream: TextIO, record: Dict[str, Any]) -> None:
    stream.write(json.dumps(clean_json(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [clean_json(v) for v in value if v is not None]
    return value


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object, got {type(obj).__name__}")
            yield obj


def context_key(context: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        context.get("context_id"),
        context.get("vmid"),
        context.get("el"),
        context.get("security"),
        context.get("isa"),
    )


def clean_context(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not record:
        return {}
    keys = ("context_id", "vmid", "el", "security", "isa")
    return {key: record[key] for key in keys if record.get(key) is not None}


def clean_trace_info(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not record:
        return {}
    keys = ("trace_id", "p0_key", "curr_spec_depth", "cc_threshold")
    return {key: record[key] for key in keys if record.get(key) is not None}


def module_from_record(record: Dict[str, Any]) -> Optional[str]:
    name = record.get("name")
    if isinstance(name, str) and name:
        return name

    module = record.get("module")
    if isinstance(module, str) and module:
        return module
    if isinstance(module, dict):
        name = module.get("name")
        if isinstance(name, str) and name:
            return name
        elf = module.get("elf")
        if isinstance(elf, str) and elf:
            return Path(elf).stem

    elf = record.get("elf")
    if isinstance(elf, str) and elf:
        return Path(elf).stem

    map_name = record.get("map")
    if isinstance(map_name, str) and map_name and not map_name.startswith("["):
        if record.get("elf_addr") is not None or record.get("elf_va") is not None:
            return map_name
    return None


def section_from_record(record: Dict[str, Any]) -> Optional[str]:
    section = record.get("section")
    if isinstance(section, str):
        return section
    if isinstance(section, dict):
        name = section.get("name")
        return name if isinstance(name, str) else None
    return None


def symbol_text_from_record(record: Dict[str, Any]) -> Optional[str]:
    symbol = record.get("symbol")
    if isinstance(symbol, str):
        return symbol
    if isinstance(symbol, dict):
        name = symbol.get("name")
        if not isinstance(name, str) or not name:
            return None
        offset = parse_int(record.get("symbol_offset"))
        return f"{name}+0x{offset or 0:x}"
    return None


def split_symbol(symbol: Optional[str]) -> Tuple[str, Optional[int], Optional[str]]:
    if not symbol:
        return UNKNOWN, None, None
    match = SYMBOL_RE.match(symbol)
    if not match:
        return symbol, 0, None
    name = match.group("name") or UNKNOWN
    offset = parse_int(match.group("offset"))
    relation = match.group("relation")
    return name, offset, relation


@dataclass
class InputJob:
    input_path: Path
    output_rel: Path


@dataclass
class AddressInfo:
    module: Optional[str]
    symbol: Optional[str]
    section: Optional[str]
    runtime_addr: Optional[int]
    elf_addr: Optional[int]
    reason: Optional[str]


@dataclass
class RangeInfo:
    source_event_index: int
    context: Dict[str, Any]
    trace_info: Dict[str, Any]
    module: Optional[str]
    start_symbol: Optional[str]
    end_symbol: Optional[str]
    runtime_start: Optional[int]
    runtime_end: Optional[int]
    elf_start: Optional[int]
    elf_end: Optional[int]
    instruction_count: Optional[int]
    atom: Optional[str]
    status: str
    reason: Optional[str]
    waypoint: Dict[str, Any]
    next_pc: Optional[int]

    @property
    def waypoint_kind(self) -> Optional[str]:
        kind = self.waypoint.get("kind")
        return kind if isinstance(kind, str) else None

    @property
    def taken(self) -> Optional[bool]:
        value = self.waypoint.get("taken")
        if isinstance(value, bool):
            return value
        kind = self.waypoint_kind
        if kind == "conditional_branch":
            if self.atom == "E":
                return True
            if self.atom == "N":
                return False
            return None
        if kind in {"direct_branch", "direct_call", "indirect_branch", "indirect_call", "return"}:
            return self.atom == "E" if self.atom in {"E", "N"} else True
        return None


@dataclass
class SegmentState:
    segment_id: int
    context: Dict[str, Any]
    trace_info: Dict[str, Any]
    start_event_index: int
    start: Optional[Dict[str, Any]] = None
    range_count: int = 0
    edge_count: int = 0


class FlowPreprocessor:
    def __init__(
        self,
        *,
        sequence_stream: TextIO,
        include_module_in_token: bool,
        target_modules: Optional[Set[str]],
        min_segment_edges: int,
    ) -> None:
        self.sequence_stream = sequence_stream
        self.include_module_in_token = include_module_in_token
        self.target_modules = target_modules
        self.min_segment_edges = max(1, min_segment_edges)

        self.stats: Counter[str] = Counter()
        self.edge_types: Counter[str] = Counter()
        self.waypoint_kinds: Counter[str] = Counter()
        self.statuses: Counter[str] = Counter()
        self.break_reasons: Counter[str] = Counter()
        self.segment_lengths: List[int] = []

        self.global_segment_index = 0
        self.global_edge_index = 0

        self.current_context: Dict[str, Any] = {}
        self.current_trace_info: Dict[str, Any] = {}
        self.default_module: Optional[str] = None
        self.current_module: Optional[str] = None
        self.address_by_runtime: Dict[int, AddressInfo] = {}
        self.address_by_elf: Dict[int, AddressInfo] = {}

        self.pending_range: Optional[RangeInfo] = None
        self.segment: Optional[SegmentState] = None
        self.segment_tokens: List[Dict[str, Any]] = []
    def process_file(self, path: Path) -> None:
        last_event_index: Optional[int] = None
        for event_index, record in enumerate(iter_jsonl(path)):
            last_event_index = event_index
            self.process_record(path, event_index, record)
        self.break_segment("file_end", last_event_index)

    def process_record(self, path: Path, event_index: int, record: Dict[str, Any]) -> None:
        event = record.get("event")
        if isinstance(event, str):
            self.stats[f"input_event_{event}"] += 1
        else:
            self.stats["input_event_missing"] += 1

        if event == "header":
            self.handle_header(record)
        elif event == "trace_stream":
            self.handle_trace_stream(record, event_index)
        elif event == "trace_info":
            self.current_trace_info = clean_trace_info(record)
        elif event == "context":
            self.handle_context(record, event_index)
        elif event == "address":
            self.handle_address(record, event_index)
        elif event == "instruction_range":
            self.handle_instruction_range(record, event_index)
        elif event == "flow_gap":
            reason = str(record.get("reason") or "flow_gap")
            self.break_segment(f"flow_gap:{reason}", event_index)
        elif event == "discontinuity":
            reason = str(record.get("reason") or "discontinuity")
            self.break_segment(f"discontinuity:{reason}", event_index)
        elif event == "trace_control":
            kind = str(record.get("kind") or record.get("reason") or "trace_control")
            if kind != "conditional_flush":
                self.break_segment(f"trace_control:{kind}", event_index)
        elif event == "overflow":
            self.break_segment("overflow", event_index)
        elif event in {"exception", "exception_return"}:
            self.break_segment(event, event_index)
        else:
            self.stats[f"ignored_event_{event or 'unknown'}"] += 1

    def break_segment(
        self,
        reason: str,
        event_index: Optional[int],
        *,
        emit_pending: bool = True,
    ) -> None:
        if emit_pending:
            self.emit_pending_edge(None)
        elif self.pending_range is not None:
            self.stats["pending_edges_dropped"] += 1
            self.stats[f"pending_edges_dropped_{reason}"] += 1
            self.pending_range = None
        self.close_segment(reason, event_index)

    def handle_header(self, record: Dict[str, Any]) -> None:
        modules = record.get("modules")
        if not isinstance(modules, list):
            return
        module_names = []
        for module_record in modules:
            if isinstance(module_record, dict):
                module = module_from_record(module_record)
                if module:
                    module_names.append(module)
        if len(module_names) == 1:
            self.default_module = module_names[0]
            self.current_module = module_names[0]

    def handle_trace_stream(self, record: Dict[str, Any], event_index: int) -> None:
        trace_id = record.get("trace_id")
        if trace_id != self.current_context.get("trace_id"):
            self.break_segment("trace_stream_change", event_index)
        if trace_id is not None:
            self.current_context["trace_id"] = trace_id

    def handle_context(self, record: Dict[str, Any], event_index: int) -> None:
        new_context = clean_context(record)
        if context_key(new_context) != context_key(self.current_context):
            self.break_segment("context_change", event_index)
        self.current_context = new_context

    def handle_address(self, record: Dict[str, Any], event_index: int) -> None:
        runtime = parse_int(record.get("address") or record.get("runtime_addr"))
        elf = parse_int(record.get("elf_addr") or record.get("elf_va"))
        reason = record.get("reason") if isinstance(record.get("reason"), str) else None
        info = AddressInfo(
            module=module_from_record(record),
            symbol=symbol_text_from_record(record),
            section=section_from_record(record),
            runtime_addr=runtime,
            elf_addr=elf,
            reason=reason,
        )
        if info.module is None and elf is not None:
            info.module = self.current_module or self.default_module

        if reason in CUT_ADDRESS_REASONS:
            pending_targets_this_address = (
                self.pending_range is not None
                and runtime is not None
                and self.pending_range.next_pc == runtime
            )
            self.break_segment(
                f"address:{reason}",
                event_index,
                emit_pending=not pending_targets_this_address,
            )
            self.stats[f"address_cut_{reason}"] += 1
            return

        if self.target_modules is not None:
            if info.module is None or info.module not in self.target_modules:
                pending_targets_this_address = (
                    self.pending_range is not None
                    and runtime is not None
                    and self.pending_range.next_pc == runtime
                )
                self.break_segment(
                    "address:outside_target_module",
                    event_index,
                    emit_pending=not pending_targets_this_address,
                )
                self.stats["address_outside_target_module"] += 1
                return

        if runtime is not None:
            self.address_by_runtime[runtime] = info
        if elf is not None:
            self.address_by_elf[elf] = info
        if info.module:
            self.current_module = info.module

    def handle_instruction_range(self, record: Dict[str, Any], event_index: int) -> None:
        item = self.normalize_range(record, event_index)
        status = item.status or "resolved"
        self.statuses[status] += 1
        self.waypoint_kinds[item.waypoint_kind or UNKNOWN] += 1

        if item.reason in CUT_ADDRESS_REASONS:
            self.break_segment(f"range:{item.reason}", event_index)
            self.stats["range_skipped_non_target"] += 1
            return

        if status == "conflict":
            self.break_segment("range_status:conflict", event_index)
            self.stats["range_skipped_conflict"] += 1
            return

        if status not in NORMAL_RANGE_STATUSES:
            self.break_segment(f"range_status:{status}", event_index)
            self.stats["range_skipped_abnormal_status"] += 1
            return

        if item.runtime_start is None or item.elf_start is None:
            self.break_segment("range_missing_start", event_index)
            self.stats["range_skipped_missing_start"] += 1
            return

        if item.runtime_end is None or item.elf_end is None:
            self.break_segment("range_missing_end", event_index)
            self.stats["range_skipped_missing_end"] += 1
            return

        if self.target_modules is not None:
            if item.module is None or item.module not in self.target_modules:
                pending_targets_this_range = (
                    self.pending_range is not None
                    and self.pending_range.next_pc == item.runtime_start
                )
                self.break_segment(
                    "range:outside_target_module",
                    event_index,
                    emit_pending=not pending_targets_this_range,
                )
                self.stats["range_skipped_outside_target_module"] += 1
                return

        if self.pending_range is None:
            self.start_segment(item)
            self.pending_range = item
            return

        if context_key(item.context) != context_key(self.pending_range.context):
            self.break_segment("context_change", event_index)
            self.start_segment(item)
            self.pending_range = item
            return

        if self.pending_range.next_pc is None:
            self.stats["pending_edges_dropped_missing_next_pc"] += 1
            self.break_segment("missing_prev_next_pc", event_index, emit_pending=False)
            self.start_segment(item)
            self.pending_range = item
            return

        if self.pending_range.next_pc != item.runtime_start:
            self.emit_pending_edge(None)
            self.close_segment("pc_mismatch", event_index)
            self.stats["pc_mismatch"] += 1
            self.start_segment(item)
            self.pending_range = item
            return

        self.emit_pending_edge(item)
        if self.segment is None:
            self.start_segment(item)
        self.pending_range = item

    def normalize_range(self, record: Dict[str, Any], event_index: int) -> RangeInfo:
        nested_context = record.get("context") if isinstance(record.get("context"), dict) else None
        if nested_context:
            context = clean_context(nested_context)
            if context_key(context) != context_key(self.current_context):
                self.break_segment("context_change", event_index)
                self.current_context = context
        else:
            context = dict(self.current_context)

        runtime_start = parse_int(record.get("runtime_start"))
        runtime_end = parse_int(record.get("runtime_end"))
        elf_start = parse_int(record.get("elf_start") or record.get("elf_va"))
        elf_end = parse_int(record.get("elf_end"))
        waypoint = record.get("waypoint") if isinstance(record.get("waypoint"), dict) else {}
        next_pc = parse_int(record.get("next_pc"))

        start_info = self.lookup_address(runtime_start, elf_start)
        end_info = self.lookup_address(runtime_end, elf_end)
        module = module_from_record(record)
        if module is None:
            module = (start_info.module if start_info else None) or (end_info.module if end_info else None)
        if module is None:
            module = self.current_module or self.default_module
        if module:
            self.current_module = module

        start_symbol = record.get("start_symbol") if isinstance(record.get("start_symbol"), str) else None
        end_symbol = record.get("end_symbol") if isinstance(record.get("end_symbol"), str) else None
        if start_symbol is None and start_info is not None:
            start_symbol = start_info.symbol
        if end_symbol is None and end_info is not None:
            end_symbol = end_info.symbol

        status = record.get("status") if isinstance(record.get("status"), str) else "resolved"
        reason = record.get("reason") if isinstance(record.get("reason"), str) else None

        return RangeInfo(
            source_event_index=event_index,
            context=context,
            trace_info=dict(self.current_trace_info),
            module=module,
            start_symbol=start_symbol,
            end_symbol=end_symbol,
            runtime_start=runtime_start,
            runtime_end=runtime_end,
            elf_start=elf_start,
            elf_end=elf_end,
            instruction_count=record.get("instruction_count")
            if isinstance(record.get("instruction_count"), int)
            else None,
            atom=record.get("atom") if isinstance(record.get("atom"), str) else None,
            status=status,
            reason=reason,
            waypoint=waypoint,
            next_pc=next_pc,
        )

    def lookup_address(self, runtime_addr: Optional[int], elf_addr: Optional[int]) -> Optional[AddressInfo]:
        if runtime_addr is not None and runtime_addr in self.address_by_runtime:
            return self.address_by_runtime[runtime_addr]
        if elf_addr is not None and elf_addr in self.address_by_elf:
            return self.address_by_elf[elf_addr]
        return None

    def start_segment(self, item: RangeInfo) -> None:
        if self.segment is not None:
            return
        segment_id = self.global_segment_index
        self.global_segment_index += 1
        self.segment = SegmentState(
            segment_id=segment_id,
            context=dict(item.context),
            trace_info=dict(item.trace_info),
            start_event_index=item.source_event_index,
        )
        self.segment_tokens.clear()

    def close_segment(self, reason: str, event_index: Optional[int]) -> None:
        had_segment = self.segment is not None
        if self.segment is not None:
            segment = self.segment
            if segment.edge_count >= self.min_segment_edges:
                self.segment_lengths.append(segment.edge_count)
                write_json(
                    self.sequence_stream,
                    {
                        "segment_id": segment.segment_id,
                        "context": segment.context,
                        "trace_info": segment.trace_info,
                        "length": len(self.segment_tokens),
                        "start": segment.start,
                        "tokens": list(self.segment_tokens),
                        "break_reason": reason,
                    },
                )
            else:
                self.stats["segments_dropped_too_short"] += 1

        if had_segment:
            self.break_reasons[reason] += 1
        self.segment = None
        self.pending_range = None
        self.segment_tokens.clear()

    def make_node(
        self,
        *,
        module: Optional[str],
        symbol: Optional[str],
        elf_addr: Optional[int],
        runtime_addr: Optional[int],
    ) -> Dict[str, Any]:
        function, func_off, _ = split_symbol(symbol)
        return {
            "module": module or UNKNOWN,
            "runtime_addr": hex_or_none(runtime_addr, width=16),
            "function": function,
            "function_offset": hex_or_none(func_off),
            "elf_addr": hex_or_none(elf_addr),
        }

    def make_start_node(self, item: RangeInfo) -> Dict[str, Any]:
        return self.make_node(
            module=item.module,
            symbol=item.start_symbol,
            elf_addr=item.elf_start,
            runtime_addr=item.runtime_start,
        )

    def make_exit_node(self, item: RangeInfo) -> Dict[str, Any]:
        return self.make_node(
            module=item.module,
            symbol=item.end_symbol,
            elf_addr=item.elf_end,
            runtime_addr=item.runtime_end,
        )

    def make_entry_node(
        self, item: RangeInfo, target_hint: Optional[RangeInfo]
    ) -> Dict[str, Any]:
        runtime_addr = item.next_pc
        symbol: Optional[str] = None
        elf_addr: Optional[int] = None
        module: Optional[str] = item.module

        if (
            target_hint is not None
            and runtime_addr is not None
            and target_hint.runtime_start == runtime_addr
        ):
            symbol = target_hint.start_symbol
            elf_addr = target_hint.elf_start
            module = target_hint.module or module
        else:
            addr_info = self.lookup_address(runtime_addr, None)
            if addr_info is not None:
                symbol = addr_info.symbol
                elf_addr = addr_info.elf_addr
                module = addr_info.module or module

        if elf_addr is None:
            elf_addr = self.derive_target_elf(item, runtime_addr)
        if symbol is None:
            symbol = self.infer_fallthrough_symbol(item, runtime_addr)

        return self.make_node(
            module=module,
            symbol=symbol,
            elf_addr=elf_addr,
            runtime_addr=runtime_addr,
        )

    def derive_target_elf(
        self, item: RangeInfo, runtime_addr: Optional[int]
    ) -> Optional[int]:
        if runtime_addr is None or item.runtime_start is None or item.elf_start is None:
            return None
        return item.elf_start + (runtime_addr - item.runtime_start)

    def infer_fallthrough_symbol(
        self, item: RangeInfo, runtime_addr: Optional[int]
    ) -> Optional[str]:
        if runtime_addr is None or item.runtime_end is None:
            return None
        if runtime_addr != item.runtime_end + 4:
            return None
        function, offset, relation = split_symbol(item.end_symbol)
        if function == UNKNOWN or offset is None:
            return None
        return f"{function}{relation or '+'}0x{offset + 4:x}"

    def edge_control(self, item: RangeInfo) -> str:
        kind = item.waypoint_kind or UNKNOWN
        taken = item.taken
        if kind == "conditional_branch":
            if taken is True:
                return "cond:T"
            if taken is False:
                return "cond:N"
            return "cond:?"
        if kind == "return":
            return "ret"
        if kind == "indirect_branch":
            return "ibranch"
        if kind == "indirect_call":
            return "icall"
        if kind == "direct_call":
            return "call"
        if kind == "direct_branch":
            return "branch"
        return kind

    def waypoint_asm(self, item: RangeInfo) -> str:
        waypoint = item.waypoint
        asm = waypoint.get("asm")
        if isinstance(asm, str) and asm:
            return asm
        text = " ".join(
            str(part)
            for part in (waypoint.get("mnemonic"), waypoint.get("op_str"))
            if part
        )
        return text or UNKNOWN

    def token_location(self, node: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "runtime_addr": node.get("runtime_addr") or UNKNOWN,
            "elf_addr": node.get("elf_addr") or UNKNOWN,
            "function": node.get("function") or UNKNOWN,
            "function_offset": node.get("function_offset") or UNKNOWN,
        }

    def make_structured_token(
        self,
        item: RangeInfo,
        exit_node: Dict[str, Any],
        entry_node: Dict[str, Any],
    ) -> Dict[str, Any]:
        token = {
            "src_ctrl": {
                **self.token_location(exit_node),
                "asm": self.waypoint_asm(item),
            },
            "ctrl": {
                "kind": item.waypoint_kind or UNKNOWN,
                "atom": item.atom or UNKNOWN,
            },
            "dst": self.token_location(entry_node),
            "instruction_count": item.instruction_count
            if item.instruction_count is not None
            else UNKNOWN,
        }
        if self.include_module_in_token:
            token["src_ctrl"]["module"] = exit_node.get("module") or UNKNOWN
            token["dst"]["module"] = entry_node.get("module") or UNKNOWN
        return token

    def emit_pending_edge(self, target_hint: Optional[RangeInfo]) -> None:
        if self.pending_range is None:
            return
        item = self.pending_range
        if item.next_pc is None:
            self.stats["pending_edges_dropped_missing_next_pc"] += 1
            self.pending_range = None
            return
        if self.segment is None:
            self.start_segment(item)
        self.emit_edge(item, target_hint)
        self.pending_range = None

    def emit_edge(self, item: RangeInfo, target_hint: Optional[RangeInfo]) -> None:
        assert self.segment is not None
        control = self.edge_control(item)
        start_node = self.make_start_node(item)
        exit_node = self.make_exit_node(item)
        entry_node = self.make_entry_node(item, target_hint)
        if self.segment.start is None:
            self.segment.start = self.token_location(start_node)
            if self.include_module_in_token:
                self.segment.start["module"] = start_node.get("module") or UNKNOWN
        token = self.make_structured_token(item, exit_node, entry_node)
        self.edge_types[control] += 1
        self.stats["output_edges"] += 1
        self.segment.edge_count += 1
        self.segment.range_count += 1
        self.global_edge_index += 1
        self.segment_tokens.append(token)

    def stats_record(self) -> Dict[str, Any]:
        length_stats: Dict[str, Any]
        if self.segment_lengths:
            sorted_lengths = sorted(self.segment_lengths)
            total_length = sum(sorted_lengths)
            length_stats = {
                "count": len(sorted_lengths),
                "min": sorted_lengths[0],
                "max": sorted_lengths[-1],
                "mean": total_length / len(sorted_lengths),
                "total": total_length,
            }
        else:
            length_stats = {"count": 0, "total": 0}

        return {
            "counts": dict(sorted(self.stats.items())),
            "edge_types": dict(sorted(self.edge_types.items())),
            "waypoint_kinds": dict(sorted(self.waypoint_kinds.items())),
            "statuses": dict(sorted(self.statuses.items())),
            "break_reasons": dict(sorted(self.break_reasons.items())),
            "segment_lengths": length_stats,
        }


def discover_inputs(paths: Sequence[Path], patterns: Sequence[str], recursive: bool) -> List[InputJob]:
    jobs: List[InputJob] = []
    seen_outputs: Set[Path] = set()

    for path_index, raw_path in enumerate(paths):
        path = raw_path.expanduser()
        if path.is_file():
            rel = unique_output_relpath(Path(path.name), seen_outputs, path_index)
            jobs.append(InputJob(path.resolve(), rel))
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"input path does not exist: {path}")

        root = path.resolve()
        matched: Dict[Path, None] = {}
        for pattern in patterns:
            iterator = root.rglob(pattern) if recursive else root.glob(pattern)
            for item in iterator:
                if item.is_file():
                    matched[item.resolve()] = None
        for item in sorted(matched.keys(), key=lambda value: str(value)):
            rel = item.relative_to(root)
            rel = unique_output_relpath(rel, seen_outputs, path_index)
            jobs.append(InputJob(item, rel))

    return jobs


def unique_output_relpath(rel: Path, seen_outputs: Set[Path], path_index: int) -> Path:
    normalized = Path(*rel.parts)
    if normalized not in seen_outputs:
        seen_outputs.add(normalized)
        return normalized

    candidate = Path(f"input_{path_index:04d}") / normalized
    counter = 1
    while candidate in seen_outputs:
        candidate = Path(f"input_{path_index:04d}_{counter:03d}") / normalized
        counter += 1
    seen_outputs.add(candidate)
    return candidate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess symbolized TRBE JSONL files into structured flow sequences."
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        type=Path,
        help="Input symbolized JSONL file(s) or directory/directories.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("flow_preprocessed"),
        help="Output directory. Each output file keeps the input file name. Default: flow_preprocessed",
    )
    parser.add_argument(
        "--glob",
        dest="patterns",
        action="append",
        default=None,
        help="File pattern for directory inputs. Can be repeated. Default: *.jsonl",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directory inputs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes. 0 means auto, 1 means serial. Default: 0",
    )
    parser.add_argument(
        "--target-module",
        action="append",
        default=None,
        help="Only keep this module/map name. Can be repeated. If omitted, no module filter is applied.",
    )
    parser.add_argument(
        "--min-segment-edges",
        type=int,
        default=1,
        help="Drop segments with fewer flow edges. Default: 1",
    )
    parser.add_argument(
        "--include-module-in-token",
        action="store_true",
        help="Include module name in each structured token location. Usually unnecessary for single-ELF training.",
    )
    return parser


def process_one_file(
    input_path_text: str,
    output_path_text: str,
    *,
    include_module_in_token: bool,
    min_segment_edges: int,
    target_modules: Optional[Set[str]],
) -> Dict[str, Any]:
    input_path = Path(input_path_text)
    output_path = Path(output_path_text)
    if output_path.resolve() == input_path.resolve():
        raise ValueError(f"refusing to overwrite input file: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sequence_stream:
        preprocessor = FlowPreprocessor(
            sequence_stream=sequence_stream,
            include_module_in_token=include_module_in_token,
            target_modules=target_modules,
            min_segment_edges=min_segment_edges,
        )
        preprocessor.process_file(input_path)
        stats = preprocessor.stats_record()

    return {"input": str(input_path), "output": str(output_path), "stats": stats}


def resolve_workers(worker_arg: int, job_count: int) -> int:
    if job_count <= 1:
        return 1
    if worker_arg > 0:
        return min(worker_arg, job_count)
    return max(1, min(os.cpu_count() or 1, job_count, 8))


def aggregate_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Counter[str] = Counter()
    edge_types: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    break_reasons: Counter[str] = Counter()
    length_count = 0
    length_total = 0
    length_min: Optional[int] = None
    length_max: Optional[int] = None

    for result in results:
        stats = result["stats"]
        counts.update(stats.get("counts", {}))
        edge_types.update(stats.get("edge_types", {}))
        statuses.update(stats.get("statuses", {}))
        break_reasons.update(stats.get("break_reasons", {}))
        lengths = stats.get("segment_lengths", {})
        count = int(lengths.get("count") or 0)
        total = int(lengths.get("total") or 0)
        length_count += count
        length_total += total
        if count:
            item_min = int(lengths.get("min"))
            item_max = int(lengths.get("max"))
            length_min = item_min if length_min is None else min(length_min, item_min)
            length_max = item_max if length_max is None else max(length_max, item_max)

    return {
        "counts": counts,
        "edge_types": edge_types,
        "statuses": statuses,
        "break_reasons": break_reasons,
        "segment_lengths": {
            "count": length_count,
            "total": length_total,
            "min": length_min,
            "max": length_max,
            "mean": (length_total / length_count) if length_count else 0.0,
        },
    }


def print_top_counter(label: str, counter: Counter[str], limit: int = 8) -> None:
    if not counter:
        print(f"{label}: none")
        return
    print(f"{label}: " + ", ".join(f"{key}={value}" for key, value in counter.most_common(limit)))


def print_summary(
    *,
    jobs: Sequence[InputJob],
    results: Sequence[Dict[str, Any]],
    aggregate: Dict[str, Any],
    output_dir: Path,
    workers: int,
) -> None:
    counts: Counter[str] = aggregate["counts"]
    lengths = aggregate["segment_lengths"]
    print(f"inputs: {len(jobs)}")
    print(f"outputs: {len(results)} -> {output_dir}")
    print(f"workers: {workers}")
    print(f"sequences: {lengths['count']}")
    print(f"edges: {counts.get('output_edges', 0)}")
    print(
        "segment_length: "
        f"min={lengths['min']} max={lengths['max']} mean={lengths['mean']:.2f}"
    )
    print_top_counter("statuses", aggregate["statuses"])
    print_top_counter("break_reasons", aggregate["break_reasons"])
    print_top_counter("edge_types", aggregate["edge_types"])

    notable = Counter(
        {
            key: value
            for key, value in counts.items()
            if key.startswith("range_skipped")
            or key.startswith("address_cut")
            or key.startswith("pending_edges_dropped")
            or key in {"segments_dropped_too_short", "pc_mismatch"}
        }
    )
    if notable:
        print_top_counter("notable_counts", notable)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.min_segment_edges <= 0:
        parser.error("--min-segment-edges must be > 0")

    patterns = args.patterns or ["*.jsonl"]
    jobs = discover_inputs(args.input, patterns, args.recursive)
    if not jobs:
        parser.error("no input files found")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_modules = set(args.target_module) if args.target_module else None
    workers = resolve_workers(args.workers, len(jobs))
    task_args = [
        (
            str(job.input_path),
            str((output_dir / job.output_rel).resolve()),
        )
        for job in jobs
    ]

    if workers == 1:
        results = [
            process_one_file(
                input_path,
                output_path,
                include_module_in_token=args.include_module_in_token,
                min_segment_edges=args.min_segment_edges,
                target_modules=target_modules,
            )
            for input_path, output_path in task_args
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    process_one_file,
                    input_path,
                    output_path,
                    include_module_in_token=args.include_module_in_token,
                    min_segment_edges=args.min_segment_edges,
                    target_modules=target_modules,
                )
                for input_path, output_path in task_args
            ]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["input"])

    aggregate = aggregate_results(results)
    print_summary(
        jobs=jobs,
        results=results,
        aggregate=aggregate,
        output_dir=output_dir,
        workers=workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
