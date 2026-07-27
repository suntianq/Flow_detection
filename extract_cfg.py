"""Extract typed static control-flow information from a Ghidra program.

The primary edge granularity is:

    control-transfer instruction address -> destination instruction address

Basic-block and function metadata are emitted as separate tables so a dynamic
ETM address can be interpreted at instruction, block, and function granularity.

This script is intended to run as a Ghidra/PyGhidra postScript.  The first
script argument is the output directory; the optional second argument is
``fast`` or ``normal``.  Fast mode skips decompiler jump-table recovery.
"""

from __future__ import print_function

import csv
import json
import os
import sys
import time
import traceback

from ghidra.app.decompiler import DecompInterface
from ghidra.program.model.block import BasicBlockModel
from ghidra.program.model.pcode import PcodeOp
from ghidra.util.task import ConsoleTaskMonitor


CONFIG = {
    "out_dir": "/home/ali/workspace/static_analysis_test",
    "fast_mode": False,
    "decomp_timeout": 30,
    "progress_interval": 50,
    "progress_log": "analysis_progress.log",
    # Canonical, typed outputs.
    "functions_csv": "functions.csv",
    "blocks_csv": "basic_blocks.csv",
    "instructions_csv": "instructions.csv",
    "memory_blocks_csv": "memory_blocks.csv",
    "cfg_edges_csv": "cfg_edges.csv",
    "call_sites_csv": "call_sites.csv",
    "call_graph_csv": "call_graph.csv",
    "indirect_detail_csv": "cfg_indirect_sites.csv",
    "return_sites_csv": "return_sites.csv",
    "errors_csv": "analysis_errors.csv",
    "metadata_json": "metadata.json",
    "summary_json": "analysis_summary.json",
}


try:
    SCRIPT_MONITOR = monitor
except NameError:
    SCRIPT_MONITOR = ConsoleTaskMonitor()


def log(message, log_fp=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(timestamp, message)
    print(line)
    if log_fp is not None:
        log_fp.write(line + "\n")
        log_fp.flush()


def safe_call(obj, method_name, default=None):
    """Call a zero-argument Java/Python method without hiding analysis errors."""
    try:
        return getattr(obj, method_name)()
    except Exception:
        return default


def java_iter(iterator):
    """Iterate either a Java iterator or a Python iterable."""
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


def address_text(address):
    return "" if address is None else str(address)


def instruction_text(instruction):
    if instruction is None:
        return ""
    try:
        return instruction.toString()
    except Exception:
        return str(instruction)


def bool_int(value):
    return 1 if value else 0


def method_bool(obj, method_name):
    try:
        return bool(getattr(obj, method_name)())
    except Exception:
        return False


def flow_properties(flow):
    return {
        "call": method_bool(flow, "isCall"),
        "jump": method_bool(flow, "isJump"),
        "conditional": method_bool(flow, "isConditional"),
        "computed": method_bool(flow, "isComputed"),
        "terminal": method_bool(flow, "isTerminal"),
        "fallthrough": method_bool(flow, "isFallthrough"),
    }


def address_offset(address, image_base):
    if address is None or image_base is None:
        return ""
    try:
        value = int(address.subtract(image_base))
        if value < 0:
            return "-0x{:x}".format(-value)
        return "0x{:x}".format(value)
    except Exception:
        return ""


def get_block_start(block):
    if block is None:
        return None
    start = safe_call(block, "getFirstStartAddress")
    if start is not None:
        return start
    return safe_call(block, "getMinAddress")


def detect_return_instruction(instruction):
    """Return ``(return_kind, detection_method)`` for one instruction."""
    if instruction is None:
        return "", ""

    mnemonic = str(safe_call(instruction, "getMnemonicString", "")).upper()
    if mnemonic in ("IRET", "IRETD", "IRETQ", "ERET", "RFI", "RFID"):
        return "exception_return", "exception_mnemonic"

    try:
        try:
            pcode_ops = instruction.getPcode(True)
        except Exception:
            pcode_ops = instruction.getPcode()
        for op in pcode_ops:
            if op.getOpcode() == PcodeOp.RETURN:
                return "return", "pcode"
    except Exception:
        pass

    if mnemonic in ("RET", "RETN", "RETF"):
        return "return", "return_mnemonic"
    # These are only fallbacks when a processor implementation does not emit a
    # RETURN p-code op.  Operand checks avoid treating every BX/JR/POP as return.
    if mnemonic in ("BX", "BR", "JR", "POP", "MOV"):
        operands = []
        try:
            for operand_index in range(instruction.getNumOperands()):
                operands.append(str(instruction.getDefaultOperandRepresentation(operand_index)).upper())
        except Exception:
            pass
        first_operand = operands[0] if operands else ""
        if mnemonic in ("BX", "BR"):
            is_return = "LR" in first_operand or "X30" in first_operand
            return ("return", "operand_heuristic") if is_return else ("", "")
        if mnemonic == "JR":
            is_return = "$RA" in first_operand or first_operand.strip() == "RA"
            return ("return", "operand_heuristic") if is_return else ("", "")
        if mnemonic == "POP":
            is_return = any("PC" in operand for operand in operands)
            return ("return", "operand_heuristic") if is_return else ("", "")
        if mnemonic == "MOV" and len(operands) >= 2:
            is_return = (
                "PC" in operands[0]
                and ("LR" in operands[1] or "X30" in operands[1])
            )
            return ("return", "operand_heuristic") if is_return else ("", "")
    return "", ""


def get_script_arguments():
    """Use Ghidra postScript arguments, with sys.argv as an external fallback."""
    try:
        return list(getScriptArgs())
    except Exception:
        return sys.argv[1:]


def program_metadata(program):
    language = safe_call(program, "getLanguage")
    compiler_spec = safe_call(program, "getCompilerSpec")
    image_base = safe_call(program, "getImageBase")
    executable_sha256 = safe_call(program, "getExecutableSHA256", "") or ""
    executable_md5 = safe_call(program, "getExecutableMD5", "") or ""
    return {
        "program_name": str(safe_call(program, "getName", "")),
        "executable_path": str(safe_call(program, "getExecutablePath", "")),
        "executable_format": str(safe_call(program, "getExecutableFormat", "")),
        "sha256": str(executable_sha256),
        "md5": str(executable_md5),
        "image_base": address_text(image_base),
        "language_id": str(safe_call(language, "getLanguageID", "")),
        "processor": str(safe_call(language, "getProcessor", "")),
        "big_endian": bool_int(method_bool(language, "isBigEndian")),
        "pointer_size": safe_call(program, "getDefaultPointerSize", ""),
        "compiler_spec": str(safe_call(compiler_spec, "getCompilerSpecID", "")),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema_version": 3,
        "edge_granularity": "control_transfer_instruction_to_destination_instruction",
    }


def classify_edge(flow, source_instruction, destination, is_return=False):
    properties = flow_properties(flow)
    source_flow = safe_call(source_instruction, "getFlowType")
    source_properties = flow_properties(source_flow)
    if is_return:
        return "return"
    # A CodeBlockReference for the continuation of a call/conditional branch
    # may itself only be typed FALL_THROUGH.  Preserve the source instruction's
    # stronger semantics so the same transition is not emitted twice with two
    # different meanings.
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
            fallthrough = safe_call(source_instruction, "getFallThrough")
            if fallthrough is not None and destination == fallthrough:
                return "conditional_not_taken"
            return "conditional_taken"
        return "indirect_jump" if properties["computed"] else "direct_jump"
    return "unknown"


def confidence_rank(confidence):
    return {"low": 1, "medium": 2, "high": 3}.get(confidence, 0)


class AnalysisResult(object):
    def __init__(self, program, block_model, listing, monitor_obj, metadata):
        self.program = program
        self.listing = listing
        self.block_model = block_model
        self.monitor = monitor_obj
        self.metadata = metadata
        self.image_base = safe_call(program, "getImageBase")
        self.function_manager = program.getFunctionManager()

        self.functions = set()
        self.blocks = set()
        self.instructions = set()
        self.memory_blocks = set()
        self.edges = {}
        self.call_sites = {}
        self.call_graph = {}
        self.indirect_sites = {}
        self.return_sites = {}
        self.errors = []
        self._block_cache = {}

        self.stats = {
            "functions_total": 0,
            "functions_completed": 0,
            "functions_failed": 0,
            "blocks_total": 0,
            "instructions_total": 0,
            "edges_total": 0,
            "decompile_attempts": 0,
            "decompile_failures": 0,
            "indirect_sites_total": 0,
            "indirect_resolved": 0,
            "indirect_partial": 0,
            "indirect_unresolved": 0,
            "return_sites_total": 0,
        }

        try:
            for memory_block in program.getMemory().getBlocks():
                start = memory_block.getStart()
                end = memory_block.getEnd()
                self.memory_blocks.add((
                    str(safe_call(memory_block, "getName", "")),
                    address_text(start),
                    address_text(end),
                    address_offset(start, self.image_base),
                    address_offset(end, self.image_base),
                    bool_int(method_bool(memory_block, "isRead")),
                    bool_int(method_bool(memory_block, "isWrite")),
                    bool_int(method_bool(memory_block, "isExecute")),
                    bool_int(method_bool(memory_block, "isInitialized")),
                    bool_int(method_bool(memory_block, "isVolatile")),
                    bool_int(method_bool(memory_block, "isOverlay")),
                ))
        except Exception as error:
            self.record_error("memory_blocks", error=error)

    def record_error(self, phase, function_entry="", block_start="", instruction_addr="", error=None):
        error_type = "" if error is None else type(error).__name__
        message = "" if error is None else str(error)
        self.errors.append((phase, function_entry, block_start, instruction_addr, error_type, message))

    def function_entry_for(self, address):
        if address is None:
            return ""
        function = None
        try:
            function = self.function_manager.getFunctionContaining(address)
        except Exception:
            pass
        if function is None:
            try:
                function = self.function_manager.getFunctionAt(address)
            except Exception:
                pass
        return address_text(safe_call(function, "getEntryPoint"))

    def block_start_for(self, address):
        if address is None:
            return ""
        key = address_text(address)
        if key in self._block_cache:
            return self._block_cache[key]
        block_start = ""
        try:
            block = self.block_model.getFirstCodeBlockContaining(address, self.monitor)
            block_start = address_text(get_block_start(block))
        except Exception:
            pass
        self._block_cache[key] = block_start
        return block_start

    def add_edge(self, src_function, src_block, src_instruction, destination, edge_type,
                 is_indirect, resolution, confidence, src_text=""):
        if src_instruction is None or destination is None or edge_type == "return":
            return
        src_addr = address_text(src_instruction)
        dst_addr = address_text(destination)
        dst_function = self.function_entry_for(destination)
        dst_block = self.block_start_for(destination)
        dst_instruction = self.listing.getInstructionAt(destination)
        dst_text = instruction_text(dst_instruction)
        interprocedural = bool(dst_function and src_function and dst_function != src_function)
        if interprocedural and edge_type in ("direct_jump", "indirect_jump"):
            edge_type = "tail_call"
        record = (
            src_function,
            dst_function,
            src_block,
            dst_block,
            src_addr,
            dst_addr,
            edge_type,
            bool_int(interprocedural),
            bool_int(is_indirect),
            resolution,
            confidence,
            src_text,
            dst_text,
        )
        key = (src_addr, dst_addr, edge_type)
        previous = self.edges.get(key)
        if previous is None or confidence_rank(confidence) > confidence_rank(previous[10]):
            self.edges[key] = record
        if edge_type == "tail_call":
            self.add_call_site(
                src_function, src_instruction, "tail_call", destination, None,
                resolution, confidence,
            )

    def add_call_site(self, caller_function, call_site, call_kind, callee, continuation,
                      resolution, confidence):
        record = (
            caller_function,
            address_text(call_site),
            call_kind,
            address_text(callee),
            address_text(continuation),
            resolution,
            confidence,
        )
        key = (address_text(call_site), address_text(callee), call_kind)
        previous = self.call_sites.get(key)
        if previous is None or confidence_rank(confidence) > confidence_rank(previous[6]):
            self.call_sites[key] = record

        callee_function = self.function_entry_for(callee)
        if callee_function:
            graph_record = (
                caller_function,
                callee_function,
                call_kind,
                resolution,
                confidence,
            )
            graph_key = (caller_function, callee_function, call_kind)
            old_graph = self.call_graph.get(graph_key)
            if old_graph is None or confidence_rank(confidence) > confidence_rank(old_graph[4]):
                self.call_graph[graph_key] = graph_record

    def add_return_site(self, function_entry, block_start, instruction,
                        return_kind, detection_method):
        site_addr = address_text(safe_call(instruction, "getAddress"))
        if not site_addr or not return_kind:
            return
        key = (site_addr, return_kind)
        self.return_sites[key] = (
            function_entry,
            block_start,
            site_addr,
            return_kind,
            detection_method,
            instruction_text(instruction),
        )

    def note_indirect_site(self, function_entry, block_start, instruction, site_kind,
                           target=None, resolution="unresolved"):
        site_addr = address_text(safe_call(instruction, "getAddress"))
        key = (site_addr, site_kind)
        state = self.indirect_sites.get(key)
        if state is None:
            state = {
                "function_entry": function_entry,
                "block_start": block_start,
                "instruction": instruction,
                "targets": set(),
                "resolutions": set(),
                "complete": False,
            }
            self.indirect_sites[key] = state
        if target is not None:
            state["targets"].add(address_text(target))
        if resolution:
            state["resolutions"].add(resolution)

    def mark_jump_table_complete(self, site_addr):
        for key, state in self.indirect_sites.items():
            if key[0] == address_text(site_addr) and key[1] == "indirect_jump":
                state["complete"] = True

    def indirect_rows(self):
        rows = []
        for key, state in self.indirect_sites.items():
            targets = sorted(state["targets"])
            if state["complete"] and targets:
                status = "resolved"
            elif targets:
                status = "partial"
            else:
                status = "unresolved"
            instruction = state["instruction"]
            rows.append((
                state["function_entry"],
                state["block_start"],
                key[0],
                key[1],
                status,
                len(targets),
                bool_int(state["complete"]),
                ";".join(targets),
                ";".join(sorted(state["resolutions"])),
                instruction_text(instruction),
            ))
        return rows


def collect_block_instructions(listing, block):
    instructions = []
    iterator = listing.getInstructions(block, True)
    for instruction in java_iter(iterator):
        instructions.append(instruction)
    return instructions


def process_instruction(result, function_entry, block_start, instruction, is_last):
    address = instruction.getAddress()
    flow = instruction.getFlowType()
    properties = flow_properties(flow)
    return_kind, return_detection = detect_return_instruction(instruction)
    is_return = bool(return_kind)
    flows = []
    try:
        flows = list(instruction.getFlows())
    except Exception:
        flows = []
    fallthrough = safe_call(instruction, "getFallThrough")
    source_text = instruction_text(instruction)

    direct_targets = [address_text(target) for target in flows]
    is_control = (
        properties["call"] or properties["jump"] or properties["computed"]
        or properties["terminal"] or is_return
    )
    result.instructions.add((
        function_entry,
        block_start,
        address_text(address),
        address_offset(address, result.image_base),
        safe_call(instruction, "getLength", ""),
        str(safe_call(instruction, "getMnemonicString", "")),
        str(flow),
        bool_int(is_control),
        bool_int(properties["conditional"]),
        bool_int(properties["computed"]),
        bool_int(fallthrough is not None),
        address_text(fallthrough),
        ";".join(sorted(direct_targets)),
        return_kind,
        source_text,
    ))

    if is_return:
        result.add_return_site(
            function_entry, block_start, instruction, return_kind, return_detection,
        )
        return True

    if properties["computed"]:
        site_kind = "indirect_call" if properties["call"] else "indirect_jump"
        result.note_indirect_site(function_entry, block_start, instruction, site_kind)
    else:
        site_kind = ""

    for destination in flows:
        edge_type = classify_edge(flow, instruction, destination)
        resolution = "ghidra_instruction_flow"
        confidence = "medium" if properties["computed"] else "high"
        result.add_edge(
            function_entry, block_start, address, destination, edge_type,
            properties["computed"], resolution, confidence, source_text,
        )
        if properties["computed"]:
            result.note_indirect_site(
                function_entry, block_start, instruction, site_kind,
                target=destination, resolution=resolution,
            )
        if properties["call"]:
            result.add_call_site(
                function_entry, address, edge_type, destination, fallthrough,
                resolution, confidence,
            )

    # Only block-boundary fallthroughs belong in the CFG.  Call and conditional
    # continuations are block boundaries even if a processor model reports them
    # unusually, so retain them explicitly.
    if fallthrough is not None and (is_last or properties["call"] or properties["conditional"]):
        if properties["call"]:
            fallthrough_type = "call_fallthrough"
        elif properties["conditional"]:
            fallthrough_type = "conditional_not_taken"
        elif not properties["jump"] and not properties["terminal"]:
            fallthrough_type = "fallthrough"
        else:
            fallthrough_type = None
        if fallthrough_type is not None:
            result.add_edge(
                function_entry, block_start, address, fallthrough, fallthrough_type,
                False, "instruction_fallthrough", "high", source_text,
            )

    if properties["call"] and not flows:
        call_kind = "indirect_call" if properties["computed"] else "direct_call"
        result.add_call_site(
            function_entry, address, call_kind, None, fallthrough,
            "unresolved", "low",
        )
    return False


def process_block_destinations(result, function_entry, block, fallback_source_instruction):
    block_start = address_text(get_block_start(block))
    try:
        destinations = block.getDestinations(result.monitor)
    except Exception as error:
        result.record_error("block_destinations", function_entry, block_start, error=error)
        return

    for reference in java_iter(destinations):
        try:
            source = reference.getSourceAddress()
            destination = reference.getDestinationAddress()
            flow = reference.getFlowType()
            source_instruction = result.listing.getInstructionAt(source)
            if source_instruction is None:
                source_instruction = fallback_source_instruction
                source = safe_call(source_instruction, "getAddress")
            if source is None or destination is None:
                continue
            properties = flow_properties(flow)
            return_kind, return_detection = detect_return_instruction(source_instruction)
            is_return = bool(return_kind)
            edge_type = classify_edge(flow, source_instruction, destination, is_return)
            if edge_type == "return":
                result.add_return_site(
                    function_entry, block_start, source_instruction,
                    return_kind, return_detection,
                )
                continue
            result.add_edge(
                function_entry,
                block_start,
                source,
                destination,
                edge_type,
                properties["computed"],
                "block_model",
                "medium" if properties["computed"] else "high",
                instruction_text(source_instruction),
            )
            if properties["computed"]:
                site_kind = "indirect_call" if properties["call"] else "indirect_jump"
                result.note_indirect_site(
                    function_entry, block_start, source_instruction, site_kind,
                    target=destination, resolution="block_model",
                )
            if properties["call"]:
                continuation = safe_call(source_instruction, "getFallThrough")
                result.add_call_site(
                    function_entry, source, edge_type, destination, continuation,
                    "block_model", "medium" if properties["computed"] else "high",
                )
        except Exception as error:
            result.record_error("block_destination", function_entry, block_start, error=error)


def recover_jump_tables(result, function, decompiler, function_entry, log_fp):
    result.stats["decompile_attempts"] += 1
    try:
        decompile_result = decompiler.decompileFunction(
            function, CONFIG["decomp_timeout"], result.monitor
        )
        if not decompile_result.decompileCompleted():
            result.stats["decompile_failures"] += 1
            result.record_error("decompile_incomplete", function_entry)
            return
        high_function = decompile_result.getHighFunction()
        if high_function is None:
            result.stats["decompile_failures"] += 1
            result.record_error("decompile_no_high_function", function_entry)
            return
        jump_tables = safe_call(high_function, "getJumpTables", []) or []
        for jump_table in jump_tables:
            switch_address = jump_table.getSwitchAddress()
            switch_instruction = result.listing.getInstructionAt(switch_address)
            block_start = result.block_start_for(switch_address)
            if switch_instruction is None:
                result.record_error(
                    "jump_table_missing_instruction", function_entry, block_start,
                    address_text(switch_address),
                )
                continue
            cases = list(jump_table.getCases())
            for target in cases:
                if target is None:
                    continue
                result.add_edge(
                    function_entry,
                    block_start,
                    switch_address,
                    target,
                    "switch_case",
                    True,
                    "decompiler_jump_table",
                    "high",
                    instruction_text(switch_instruction),
                )
                result.note_indirect_site(
                    function_entry, block_start, switch_instruction, "indirect_jump",
                    target=target, resolution="decompiler_jump_table",
                )
            result.mark_jump_table_complete(switch_address)
    except Exception as error:
        result.stats["decompile_failures"] += 1
        result.record_error("decompile_exception", function_entry, error=error)
        log("Decompiler error in function {}: {}".format(function_entry, error), log_fp)


def process_function(result, function, decompiler, fast_mode, log_fp):
    entry = function.getEntryPoint()
    function_entry = address_text(entry)
    body = function.getBody()
    result.functions.add((
        function_entry,
        address_offset(entry, result.image_base),
        str(safe_call(function, "getName", "")),
        address_text(safe_call(body, "getMinAddress")),
        address_text(safe_call(body, "getMaxAddress")),
        bool_int(method_bool(function, "isExternal")),
        bool_int(method_bool(function, "isThunk")),
    ))

    needs_decompiler = False
    blocks = result.block_model.getCodeBlocksContaining(body, result.monitor)
    for block in java_iter(blocks):
        result.monitor.checkCancelled()
        block_start_address = get_block_start(block)
        block_start = address_text(block_start_address)
        instructions = collect_block_instructions(result.listing, block)
        if not instructions:
            result.record_error("empty_block", function_entry, block_start)
            continue
        last_instruction = instructions[-1]
        result.blocks.add((
            function_entry,
            block_start,
            address_offset(block_start_address, result.image_base),
            address_text(safe_call(block, "getMaxAddress")),
            address_text(last_instruction.getAddress()),
            len(instructions),
        ))
        result.stats["blocks_total"] += 1

        for index, instruction in enumerate(instructions):
            is_return = process_instruction(
                result,
                function_entry,
                block_start,
                instruction,
                index == len(instructions) - 1,
            )
            result.stats["instructions_total"] += 1
            flow = instruction.getFlowType()
            properties = flow_properties(flow)
            if properties["computed"] and properties["jump"] and not is_return:
                needs_decompiler = True

        process_block_destinations(result, function_entry, block, last_instruction)

    if needs_decompiler and not fast_mode:
        recover_jump_tables(result, function, decompiler, function_entry, log_fp)


def analyze_program(program, fast_mode, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    log_path = os.path.join(out_dir, CONFIG["progress_log"])
    log_fp = open(log_path, "a", encoding="utf-8")

    listing = program.getListing()
    block_model = BasicBlockModel(program)
    metadata = program_metadata(program)
    result = AnalysisResult(program, block_model, listing, SCRIPT_MONITOR, metadata)
    decompiler = DecompInterface()
    if not decompiler.openProgram(program):
        log_fp.close()
        raise RuntimeError("Ghidra decompiler could not open the current program")

    functions = list(java_iter(program.getFunctionManager().getFunctions(True)))
    result.stats["functions_total"] = len(functions)
    start_time = time.time()
    log(
        "Starting typed CFG extraction: {} functions, fast_mode={}".format(
            len(functions), fast_mode
        ),
        log_fp,
    )

    try:
        for index, function in enumerate(functions, 1):
            SCRIPT_MONITOR.checkCancelled()
            function_entry = address_text(safe_call(function, "getEntryPoint"))
            try:
                process_function(result, function, decompiler, fast_mode, log_fp)
                result.stats["functions_completed"] += 1
            except Exception as error:
                result.stats["functions_failed"] += 1
                result.record_error("function", function_entry, error=error)
                log(
                    "Function {} failed: {}\n{}".format(
                        function_entry, error, traceback.format_exc()
                    ),
                    log_fp,
                )

            if index == 1 or index % CONFIG["progress_interval"] == 0 or index == len(functions):
                elapsed = time.time() - start_time
                rate = index / elapsed if elapsed > 0 else 0.0
                remaining = (len(functions) - index) / rate if rate > 0 else 0.0
                log(
                    "Progress {}/{}; elapsed {:.1f}s; ETA {:.1f}s; errors {}".format(
                        index, len(functions), elapsed, remaining, len(result.errors)
                    ),
                    log_fp,
                )
    finally:
        try:
            decompiler.dispose()
        finally:
            log_fp.close()

    indirect_rows = result.indirect_rows()
    result.stats["edges_total"] = len(result.edges)
    result.stats["indirect_sites_total"] = len(indirect_rows)
    result.stats["indirect_resolved"] = sum(1 for row in indirect_rows if row[4] == "resolved")
    result.stats["indirect_partial"] = sum(1 for row in indirect_rows if row[4] == "partial")
    result.stats["indirect_unresolved"] = sum(1 for row in indirect_rows if row[4] == "unresolved")
    result.stats["return_sites_total"] = len(result.return_sites)
    result.stats["elapsed_seconds"] = round(time.time() - start_time, 3)
    return result


def atomic_write_csv(path, header, rows):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in sorted(rows, key=lambda value: tuple(str(item) for item in value)):
            writer.writerow(row)
        csv_file.flush()
        try:
            os.fsync(csv_file.fileno())
        except Exception:
            pass
    os.replace(temporary_path, path)


def atomic_write_json(path, value):
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


def write_outputs(result, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    edge_rows = list(result.edges.values())
    call_site_rows = list(result.call_sites.values())
    call_graph_rows = list(result.call_graph.values())
    indirect_detail_rows = result.indirect_rows()
    return_site_rows = list(result.return_sites.values())

    atomic_write_csv(
        os.path.join(out_dir, CONFIG["functions_csv"]),
        ["function_entry", "function_offset", "name", "body_min", "body_max", "is_external", "is_thunk"],
        result.functions,
    )
    atomic_write_csv(
        os.path.join(out_dir, CONFIG["blocks_csv"]),
        ["function_entry", "block_start", "block_offset", "block_end", "last_instr", "instruction_count"],
        result.blocks,
    )
    atomic_write_csv(
        os.path.join(out_dir, CONFIG["instructions_csv"]),
        ["function_entry", "block_start", "instr_addr", "instr_offset", "instr_size", "mnemonic", "flow_type", "is_control", "is_conditional", "is_indirect", "has_fallthrough", "fallthrough_target", "direct_targets", "return_kind", "text"],
        result.instructions,
    )
    atomic_write_csv(
        os.path.join(out_dir, CONFIG["memory_blocks_csv"]),
        ["name", "start", "end", "start_offset", "end_offset", "is_read", "is_write", "is_execute", "is_initialized", "is_volatile", "is_overlay"],
        result.memory_blocks,
    )
    atomic_write_csv(
        os.path.join(out_dir, CONFIG["cfg_edges_csv"]),
        ["src_function", "dst_function", "src_block", "dst_block", "src_instr", "dst_instr", "edge_type", "is_interprocedural", "is_indirect", "resolution", "confidence", "src_text", "dst_text"],
        edge_rows,
    )
    atomic_write_csv(
        os.path.join(out_dir, CONFIG["call_sites_csv"]),
        ["caller_function", "call_site", "call_kind", "callee", "continuation", "resolution", "confidence"],
        call_site_rows,
    )
    atomic_write_csv(
        os.path.join(out_dir, CONFIG["call_graph_csv"]),
        ["caller_function", "callee_function", "call_kind", "resolution", "confidence"],
        call_graph_rows,
    )
    atomic_write_csv(
        os.path.join(out_dir, CONFIG["indirect_detail_csv"]),
        ["function_entry", "block_start", "site_addr", "site_kind", "resolution_status", "resolved_target_count", "is_complete", "resolved_targets", "resolutions", "text"],
        indirect_detail_rows,
    )
    atomic_write_csv(
        os.path.join(out_dir, CONFIG["return_sites_csv"]),
        ["function_entry", "block_start", "site_addr", "return_kind", "detection_method", "text"],
        return_site_rows,
    )
    atomic_write_csv(
        os.path.join(out_dir, CONFIG["errors_csv"]),
        ["phase", "function_entry", "block_start", "instruction_addr", "error_type", "message"],
        result.errors,
    )

    metadata = dict(result.metadata)
    metadata["fast_mode"] = CONFIG["fast_mode"]
    metadata["decompiler_timeout_seconds"] = CONFIG["decomp_timeout"]
    metadata["outputs"] = {
        key: value for key, value in CONFIG.items()
        if key.endswith("_csv") or key.endswith("_json")
    }
    atomic_write_json(os.path.join(out_dir, CONFIG["metadata_json"]), metadata)
    atomic_write_json(os.path.join(out_dir, CONFIG["summary_json"]), result.stats)


def main():
    try:
        program = currentProgram
    except NameError:
        program = globals().get("currentProgram")
    if program is None:
        print("Error: currentProgram is unavailable; run this as a Ghidra/PyGhidra script.")
        return 1

    arguments = get_script_arguments()
    if len(arguments) >= 1 and arguments[0]:
        CONFIG["out_dir"] = arguments[0]
    if len(arguments) >= 2:
        mode = str(arguments[1]).lower()
        CONFIG["fast_mode"] = mode in ("fast", "fast_mode", "fastmode")

    print("Configuration: {}".format(CONFIG))
    try:
        result = analyze_program(
            program, CONFIG["fast_mode"], CONFIG["out_dir"]
        )
        write_outputs(result, CONFIG["out_dir"])
        print("Wrote typed CFG outputs to {}".format(CONFIG["out_dir"]))
        print("Summary: {}".format(result.stats))
        return 0 if result.stats["functions_failed"] == 0 else 2
    except Exception as error:
        print("Analysis failed: {}".format(error))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        # Ghidra scripts do not always honor SystemExit as a process exit status,
        # but raising it still makes interactive/script failures visible.
        raise SystemExit(exit_code)
