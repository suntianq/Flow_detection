#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import concurrent.futures
import bisect
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, TextIO, Tuple

SCRIPT_VERSION = "2.6.0"

try:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection
except ImportError as exc:  # pragma: no cover - 运行环境提示
    ELFFile = None  # type: ignore[assignment]
    SymbolTableSection = None  # type: ignore[assignment]
    ELFTOOLS_IMPORT_ERROR = exc
else:
    ELFTOOLS_IMPORT_ERROR = None

# Capstone 主要用于输出可读反汇编；常见 AArch64 控制流有内置解码。
# 少数较新的 PAC 分支/返回指令仍依赖 Capstone mnemonic 做补充分流。
try:
    from capstone import (
        Cs,
        CS_ARCH_ARM64,
        CS_MODE_ARM,
        CS_MODE_BIG_ENDIAN,
        CS_MODE_LITTLE_ENDIAN,
    )
except ImportError:  # pragma: no cover - 可选依赖
    Cs = None  # type: ignore[assignment]
    CS_ARCH_ARM64 = CS_MODE_ARM = CS_MODE_BIG_ENDIAN = CS_MODE_LITTLE_ENDIAN = 0


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def parse_int(value: Any, *, field_name: str = "value") -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip(), 0)
        except ValueError as exc:
            raise ValueError(f"{field_name} 不是合法整数：{value!r}") from exc
    raise TypeError(f"{field_name} 必须是整数或字符串，实际为 {type(value).__name__}")


def parse_optional_int(value: Any, *, field_name: str = "value") -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "none"}:
        return None
    return parse_int(value, field_name=field_name)


def align_down(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment 必须大于 0")
    return value - value % alignment


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment 必须大于 0")
    return ((value + alignment - 1) // alignment) * alignment


def parse_trace_id(value: str) -> int:
    return int(value, 16) if value.lower().startswith("0x") else int(value, 10)


def permission_from_flags(flags: int) -> str:
    # ELF PF_X=1, PF_W=2, PF_R=4
    return "".join(
        (
            "r" if flags & 4 else "-",
            "w" if flags & 2 else "-",
            "x" if flags & 1 else "-",
        )
    )


def permission_compatible(map_perms: str, segment_flags: int) -> bool:
    seg_perms = permission_from_flags(segment_flags)
    # 指令地址映射时，x 是最重要的；r/w 允许 loader 后续拆分或收紧权限。
    if "x" in map_perms and "x" not in seg_perms:
        return False
    return True


class Stopwatch:
    def __init__(self) -> None:
        self._started: Dict[str, float] = {}
        self.accumulated: Dict[str, float] = defaultdict(float)

    def start(self, key: str) -> None:
        self._started[key] = time.perf_counter()

    def stop(self, key: str) -> None:
        started = self._started.pop(key, None)
        if started is not None:
            self.accumulated[key] += time.perf_counter() - started

    def summary(self) -> Dict[str, float]:
        return dict(sorted(self.accumulated.items()))


# ---------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------


MAPS_RE = re.compile(
    r"^([0-9a-fA-F]+)-([0-9a-fA-F]+)\s+"
    r"([rwxps-]{4})\s+"
    r"([0-9a-fA-F]+)\s+"
    r"(\S+)\s+"
    r"(\S+)"
    r"(?:\s+(.*))?$"
)


@dataclass(frozen=True)
class MapEntry:
    start: int
    end: int
    perms: str
    pgoff: int
    device: str
    inode: str
    path: str
    line_no: int

    @property
    def size(self) -> int:
        return self.end - self.start

    def contains(self, addr: int) -> bool:
        return self.start <= addr < self.end

    def offset_of(self, addr: int) -> int:
        return addr - self.start

    @property
    def display_name(self) -> str:
        return self.path or "[anonymous]"


def parse_maps(maps_path: Optional[str]) -> List[MapEntry]:
    if not maps_path:
        return []

    path = Path(maps_path)
    if not path.is_file():
        raise FileNotFoundError(f"maps 文件不存在：{path}")

    entries: List[MapEntry] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\n")
            match = MAPS_RE.match(line.strip())
            if match is None:
                print(
                    f"[WARN] maps 第 {line_no} 行格式无法识别，已跳过：{line}",
                    file=sys.stderr,
                )
                continue
            entries.append(
                MapEntry(
                    start=int(match.group(1), 16),
                    end=int(match.group(2), 16),
                    perms=match.group(3),
                    pgoff=int(match.group(4), 16),
                    device=match.group(5),
                    inode=match.group(6),
                    path=(match.group(7) or "").strip(),
                    line_no=line_no,
                )
            )

    entries.sort(key=lambda item: (item.start, item.end))
    return entries


class MapIndex:
    def __init__(self, entries: Sequence[MapEntry]) -> None:
        self.entries = list(entries)
        self.starts = [entry.start for entry in self.entries]

    def find(self, addr: int) -> Optional[MapEntry]:
        index = bisect.bisect_right(self.starts, addr) - 1
        if index >= 0:
            candidate = self.entries[index]
            if candidate.contains(addr):
                return candidate
        return None


# ---------------------------------------------------------------------------
# ELF 元数据
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadSegment:
    index: int
    p_offset: int
    p_vaddr: int
    p_filesz: int
    p_memsz: int
    flags: int
    align: int

    @property
    def perms(self) -> str:
        return permission_from_flags(self.flags)

    def contains_va(self, elf_va: int) -> bool:
        return self.p_vaddr <= elf_va < self.p_vaddr + self.p_memsz

    def contains_file_backed_va(self, elf_va: int) -> bool:
        return self.p_vaddr <= elf_va < self.p_vaddr + self.p_filesz

    def file_offset_for_va(self, elf_va: int) -> Optional[int]:
        if not self.contains_file_backed_va(elf_va):
            return None
        return self.p_offset + (elf_va - self.p_vaddr)


@dataclass(frozen=True)
class SectionRange:
    name: str
    addr: int
    size: int
    flags: int
    section_type: str

    def contains(self, elf_va: int) -> bool:
        return self.addr <= elf_va < self.addr + self.size


@dataclass(frozen=True)
class FunctionSymbol:
    name: str
    addr: int
    size: int
    bind: str
    symbol_type: str
    section_index: Any

    def contains(self, elf_va: int) -> bool:
        if self.size > 0:
            return self.addr <= elf_va < self.addr + self.size
        return self.addr == elf_va


class ElfImage:
    """一次性读取 ELF 的装载段、节和函数符号；DWARF 交给 addr2line。"""

    def __init__(self, path: Path, name: str) -> None:
        self.path = path.resolve()
        self.name = name
        self.elf_type = "?"
        self.machine = "?"
        self.entry = 0
        self.elf_class = 0
        self.little_endian = True
        self.load_segments: List[LoadSegment] = []
        self.sections: List[SectionRange] = []
        self.function_symbols: List[FunctionSymbol] = []
        self._function_addrs: List[int] = []
        self._blob: bytes = b""
        self._load()

    def _load(self) -> None:
        if ELFFile is None or SymbolTableSection is None:
            raise RuntimeError(
                "缺少 pyelftools，请安装：python -m pip install pyelftools"
            ) from ELFTOOLS_IMPORT_ERROR

        if not self.path.is_file():
            raise FileNotFoundError(f"ELF 文件不存在：{self.path}")

        self._blob = self.path.read_bytes()
        with self.path.open("rb") as stream:
            elf = ELFFile(stream)
            self.elf_type = str(elf.header["e_type"])
            self.machine = str(elf.header["e_machine"])
            self.entry = int(elf.header["e_entry"])
            self.elf_class = int(elf.elfclass)
            self.little_endian = bool(elf.little_endian)

            load_index = 0
            for segment in elf.iter_segments():
                if segment["p_type"] != "PT_LOAD":
                    continue
                self.load_segments.append(
                    LoadSegment(
                        index=load_index,
                        p_offset=int(segment["p_offset"]),
                        p_vaddr=int(segment["p_vaddr"]),
                        p_filesz=int(segment["p_filesz"]),
                        p_memsz=int(segment["p_memsz"]),
                        flags=int(segment["p_flags"]),
                        align=int(segment["p_align"]),
                    )
                )
                load_index += 1

            # Section Header 只用于输出节名。单个特殊节失败不应阻止主流程。
            try:
                for section in elf.iter_sections():
                    header = section.header
                    size = int(header["sh_size"])
                    addr = int(header["sh_addr"])
                    flags = int(header["sh_flags"])
                    # 只有 SHF_ALLOC 节才占用运行时虚拟地址。
                    # .debug_line/.debug_info 等 DWARF 节不是运行时代码或数据，
                    # 某些定制 ELF 的调试节 sh_addr 可能与 .text 重叠，不能参与地址归属。
                    if size <= 0 or (flags & 0x2) == 0:  # SHF_ALLOC
                        continue
                    self.sections.append(
                        SectionRange(
                            name=section.name or "<unnamed>",
                            addr=addr,
                            size=size,
                            flags=flags,
                            section_type=str(header["sh_type"]),
                        )
                    )
            except Exception as exc:
                print(
                    f"[WARN] {self.path.name} 的 Section Header 未完全解析：{exc}",
                    file=sys.stderr,
                )

            # 优先使用 .symtab，同时保留 .dynsym；去重后用于函数+偏移展示。
            symbols: Dict[Tuple[int, str], FunctionSymbol] = {}
            try:
                for section in elf.iter_sections():
                    if not isinstance(section, SymbolTableSection):
                        continue
                    for symbol in section.iter_symbols():
                        info = symbol.entry["st_info"]
                        symbol_type = str(info["type"])
                        if symbol_type not in {"STT_FUNC", "STT_GNU_IFUNC", "STT_NOTYPE"}:
                            continue
                        addr = int(symbol.entry["st_value"])
                        if addr == 0 or not symbol.name:
                            continue
                        item = FunctionSymbol(
                            name=symbol.name,
                            addr=addr,
                            size=int(symbol.entry["st_size"]),
                            bind=str(info["bind"]),
                            symbol_type=symbol_type,
                            section_index=symbol.entry["st_shndx"],
                        )
                        key = (item.addr, item.name)
                        old = symbols.get(key)
                        if old is None or item.size > old.size:
                            symbols[key] = item
            except Exception as exc:
                print(
                    f"[WARN] {self.path.name} 的符号表未完全解析：{exc}",
                    file=sys.stderr,
                )

        self.load_segments.sort(key=lambda item: item.p_vaddr)
        self.sections.sort(key=lambda item: (item.addr, item.size))
        self.function_symbols = sorted(symbols.values(), key=lambda item: (item.addr, -item.size, item.name))
        self._function_addrs = [item.addr for item in self.function_symbols]

        if not self.load_segments:
            raise ValueError(f"{self.path} 不包含 PT_LOAD，无法进行运行时地址映射")

    def segment_for_va(self, elf_va: int, *, executable_only: bool = False) -> Optional[LoadSegment]:
        for segment in self.load_segments:
            if executable_only and "x" not in segment.perms:
                continue
            if segment.contains_va(elf_va):
                return segment
        return None

    def read_bytes_for_va(self, elf_va: int, size: int) -> Optional[bytes]:
        """读取 ELF 中某个已落盘 PT_LOAD 对应的机器码。"""
        if size <= 0:
            return b""
        segment = self.segment_for_va(elf_va, executable_only=True)
        if segment is None:
            return None
        return self.read_bytes_from_segment(segment, elf_va, size)

    def read_bytes_from_segment(
        self, segment: LoadSegment, elf_va: int, size: int
    ) -> Optional[bytes]:
        if size <= 0:
            return b""
        file_offset = segment.file_offset_for_va(elf_va)
        if file_offset is None:
            return None
        segment_file_end = segment.p_offset + segment.p_filesz
        end = file_offset + size
        if end > segment_file_end or end > len(self._blob):
            return None
        return self._blob[file_offset:end]

    def section_for_va(self, elf_va: int) -> Optional[SectionRange]:
        # self.sections 已过滤为 SHF_ALLOC。若多个装载节重叠，优先选择
        # SHF_EXECINSTR 的可执行节，再选择范围更小的节。
        candidates = [section for section in self.sections if section.contains(elf_va)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                0 if (item.flags & 0x4) else 1,  # SHF_EXECINSTR
                item.size,
                item.addr,
            ),
        )

    def symbol_for_va(self, elf_va: int) -> Tuple[Optional[FunctionSymbol], Optional[int], bool]:
        if not self.function_symbols:
            return None, None, False

        index = bisect.bisect_right(self._function_addrs, elf_va) - 1
        if index < 0:
            return None, None, False

        # 相同地址可能有多个符号，向前后查看少量候选。
        start = max(0, index - 8)
        end = min(len(self.function_symbols), index + 9)
        containing: List[FunctionSymbol] = []
        preceding: List[FunctionSymbol] = []
        for symbol in self.function_symbols[start:end]:
            if symbol.addr <= elf_va:
                preceding.append(symbol)
            if symbol.contains(elf_va):
                containing.append(symbol)

        if containing:
            best = min(
                containing,
                key=lambda item: (
                    0 if item.symbol_type in {"STT_FUNC", "STT_GNU_IFUNC"} else 1,
                    item.size if item.size > 0 else 1 << 62,
                    -item.addr,
                ),
            )
            return best, elf_va - best.addr, True

        if preceding:
            best = max(preceding, key=lambda item: item.addr)
            return best, elf_va - best.addr, False

        return None, None, False


# ---------------------------------------------------------------------------
# 模块配置与 Load Bias 推断
# ---------------------------------------------------------------------------


@dataclass
class ModuleRule:
    name: str
    elf_path: Path
    load_bias: Optional[int] = None
    anchor_maps: List[str] = field(default_factory=list)
    context_ids: Optional[Set[int]] = None
    vmids: Optional[Set[int]] = None
    els: Optional[Set[str]] = None
    runtime_ranges: List[Tuple[int, int]] = field(default_factory=list)
    image: Optional[ElfImage] = None
    inferred_from: Optional[str] = None

    def context_matches(self, context: "ContextState") -> bool:
        if self.context_ids is not None and context.context_id not in self.context_ids:
            return False
        if self.vmids is not None and context.vmid not in self.vmids:
            return False
        if self.els is not None and context.el not in self.els:
            return False
        return True

    def runtime_range_matches(self, addr: int) -> bool:
        if not self.runtime_ranges:
            return True
        return any(start <= addr < end for start, end in self.runtime_ranges)

    def map_alias_matches(self, entry: Optional[MapEntry]) -> bool:
        if entry is None or not self.anchor_maps:
            return False
        return any(re.search(pattern, entry.path) for pattern in self.anchor_maps)


def parse_set_of_ints(values: Optional[Iterable[Any]], field_name: str) -> Optional[Set[int]]:
    if values is None:
        return None
    result = {parse_int(item, field_name=field_name) for item in values}
    return result or None


def resolve_config_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path)


def load_module_config(config_path: Path) -> Tuple[int, List[ModuleRule]]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    base_dir = config_path.resolve().parent
    page_size = parse_int(config.get("page_size", "0x1000"), field_name="page_size")
    modules_value = config.get("modules")
    if not isinstance(modules_value, list) or not modules_value:
        raise ValueError("module config 必须包含非空 modules 数组")

    modules: List[ModuleRule] = []
    for index, item in enumerate(modules_value):
        if not isinstance(item, dict):
            raise TypeError(f"modules[{index}] 必须是对象")
        name = str(item.get("name") or f"module_{index}")
        elf_value = item.get("elf")
        if not elf_value:
            raise ValueError(f"modules[{index}] 缺少 elf")

        ranges: List[Tuple[int, int]] = []
        for range_index, raw_range in enumerate(item.get("runtime_ranges", [])):
            if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                raise ValueError(f"{name}.runtime_ranges[{range_index}] 必须是 [start, end]")
            start = parse_int(raw_range[0], field_name=f"{name}.runtime_ranges.start")
            end = parse_int(raw_range[1], field_name=f"{name}.runtime_ranges.end")
            if end <= start:
                raise ValueError(f"{name}.runtime_ranges[{range_index}] 结束地址必须大于起始地址")
            ranges.append((start, end))

        els_raw = item.get("els")
        els = {str(value).upper() for value in els_raw} if els_raw else None
        modules.append(
            ModuleRule(
                name=name,
                elf_path=resolve_config_path(base_dir, str(elf_value)),
                load_bias=parse_optional_int(item.get("load_bias"), field_name=f"{name}.load_bias"),
                anchor_maps=[str(value) for value in item.get("anchor_maps", [])],
                context_ids=parse_set_of_ints(item.get("context_ids"), f"{name}.context_ids"),
                vmids=parse_set_of_ints(item.get("vmids"), f"{name}.vmids"),
                els=els,
                runtime_ranges=ranges,
            )
        )

    return page_size, modules


def infer_load_bias(
    module: ModuleRule,
    maps: Sequence[MapEntry],
    page_size: int,
) -> Tuple[int, str]:
    assert module.image is not None

    candidates: List[Tuple[int, int, str]] = []
    anchor_patterns = [re.compile(pattern) for pattern in module.anchor_maps]

    for mapping in maps:
        anchor_match = any(pattern.search(mapping.path) for pattern in anchor_patterns)
        basename_match = module.name.lower() in os.path.basename(mapping.path).lower()

        # 有 anchor 时，优先仅从 anchor map 推断，避免 [prehistoric] 等匿名段误匹配。
        if anchor_patterns and not anchor_match:
            continue

        for segment in module.image.load_segments:
            if not permission_compatible(mapping.perms, segment.flags):
                continue

            aligned_vaddr = align_down(segment.p_vaddr, page_size)
            aligned_offset = align_down(segment.p_offset, page_size)
            segment_runtime_size = align_up(
                (segment.p_vaddr - aligned_vaddr) + segment.p_memsz,
                page_size,
            )
            bias = mapping.start - aligned_vaddr

            score = 0
            reasons: List[str] = []
            if anchor_match:
                score += 1000
                reasons.append("anchor-map")
            if basename_match:
                score += 300
                reasons.append("elf-name")
            if mapping.pgoff == aligned_offset:
                score += 200
                reasons.append("pgoff")
            elif mapping.pgoff != 0:
                score -= 200

            if mapping.size == segment_runtime_size:
                score += 400
                reasons.append("exact-size")
            elif mapping.size <= segment_runtime_size:
                score += 80
                reasons.append("partial-size")
            else:
                score -= min(300, (mapping.size - segment_runtime_size) // max(page_size, 1))

            if ("x" in mapping.perms) == ("x" in segment.perms):
                score += 150
                reasons.append("exec-perm")

            # 以该 bias 映射全部 maps，统计有多少地址区间能落入任一 PT_LOAD。
            coverage = 0
            executable_coverage = 0
            for other in maps:
                sample_addr = other.start
                elf_va = sample_addr - bias
                matched_segment = module.image.segment_for_va(elf_va)
                if matched_segment is not None:
                    coverage += 1
                    if "x" in other.perms and "x" in matched_segment.perms:
                        executable_coverage += 1
            score += coverage * 10 + executable_coverage * 30
            reasons.append(f"coverage={coverage}/{executable_coverage}x")

            description = (
                f"map={mapping.display_name}@0x{mapping.start:x}, "
                f"PT_LOAD#{segment.index}@0x{segment.p_vaddr:x}, "
                + ",".join(reasons)
            )
            candidates.append((score, bias, description))

    if not candidates:
        hint = "；请传入 --load-bias 或在 module config 中设置 load_bias"
        if module.anchor_maps:
            hint += f"；当前 anchor_maps={module.anchor_maps!r} 未匹配"
        raise ValueError(f"无法自动推断模块 {module.name} 的 Load Bias{hint}")

    candidates.sort(key=lambda item: (item[0], -abs(item[1])), reverse=True)
    score, bias, description = candidates[0]
    return bias, f"score={score}; {description}"


# ---------------------------------------------------------------------------
# 外部 addr2line
# ---------------------------------------------------------------------------


class Addr2lineProcess:
    def __init__(self, tool_path: str, elf_path: Path) -> None:
        self.tool_path = tool_path
        self.elf_path = elf_path
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            [tool_path, "-f", "-C", "-p", "-e", str(elf_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def alive(self) -> bool:
        return self.process.poll() is None

    def query(self, elf_va: int) -> Optional[str]:
        with self.lock:
            if not self.alive() or self.process.stdin is None or self.process.stdout is None:
                return None
            try:
                self.process.stdin.write(f"0x{elf_va:x}\n")
                self.process.stdin.flush()
                line = self.process.stdout.readline()
            except (BrokenPipeError, OSError):
                return None

        result = line.strip()
        if not result or result in {"?? ??:0", "?? at ??:0"}:
            return None
        return result

    def close(self) -> None:
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait(timeout=1)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


class Addr2lineManager:
    def __init__(self, tool_path: Optional[str], stopwatch: Stopwatch) -> None:
        self.tool_path = tool_path
        self.stopwatch = stopwatch
        self.processes: Dict[Path, Addr2lineProcess] = {}
        self.cache: Dict[Tuple[Path, int], Optional[str]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def query(self, elf_path: Path, elf_va: int) -> Optional[str]:
        if not self.tool_path:
            return None
        key = (elf_path, elf_va)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        self.cache_misses += 1
        process = self.processes.get(elf_path)
        if process is None or not process.alive():
            try:
                process = Addr2lineProcess(self.tool_path, elf_path)
            except OSError as exc:
                print(f"[WARN] 无法启动 {self.tool_path}: {exc}", file=sys.stderr)
                self.cache[key] = None
                return None
            self.processes[elf_path] = process

        self.stopwatch.start("addr2line")
        result = process.query(elf_va)
        self.stopwatch.stop("addr2line")
        self.cache[key] = result
        return result

    def close(self) -> None:
        for process in self.processes.values():
            process.close()


def resolve_addr2line(custom_path: Optional[str]) -> Optional[str]:
    if custom_path:
        path = Path(custom_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"addr2line 不存在：{path}")
        return str(path.resolve())

    # 优先 LLVM：对 AArch64、DWARF5 和定制工具链通常更稳妥。
    for name in ("llvm-addr2line", "addr2line", "aarch64-linux-gnu-addr2line"):
        found = shutil.which(name)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# 地址解析结果
# ---------------------------------------------------------------------------


@dataclass
class ContextState:
    trace_id: Optional[int] = None
    context_id: Optional[int] = None
    vmid: Optional[int] = None
    el: Optional[str] = None
    security: Optional[str] = None
    isa: Optional[str] = None
    timestamp: Optional[int] = None
    generation: int = 0
    in_exception: bool = False

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)

    def reset_execution_context(self) -> None:
        self.context_id = None
        self.vmid = None
        self.el = None
        self.security = None
        self.isa = None
        self.in_exception = False


@dataclass
class Resolution:
    runtime_addr: int
    mapping: Optional[MapEntry]
    module: Optional[ModuleRule]
    elf_va: Optional[int]
    file_offset: Optional[int]
    segment: Optional[LoadSegment]
    section: Optional[SectionRange]
    symbol: Optional[FunctionSymbol]
    symbol_offset: Optional[int]
    symbol_contains: bool
    source: Optional[str]
    reason: Optional[str]


@dataclass
class FastResolution:
    runtime_addr: int
    mapping: Optional[MapEntry]
    module: Optional[ModuleRule]
    elf_va: Optional[int]
    segment: Optional[LoadSegment]
    raw: Optional[bytes]
    reason: Optional[str]


@dataclass
class PendingAddressTarget:
    """等待下一个 Address 包补全的动态控制流目标。"""
    source_pc: int
    source_elf_va: Optional[int]
    source_symbol: Optional[str]
    kind: str
    mnemonic: str
    context_generation: int
    trace_id: Optional[int]
    context_id: Optional[int]
    vmid: Optional[int]
    el: Optional[str]
    security: Optional[str]
    timestamp: Optional[int]
    return_address: Optional[int] = None
    range_context: Optional[ContextState] = None
    range_start_resolution: Optional[Resolution] = None
    range_end_resolution: Optional[Resolution] = None
    range_waypoint: Optional["DecodedInstruction"] = None
    range_instruction_count: int = 0
    range_atom: Optional[str] = None
    range_taken: Optional[bool] = None
    range_reason: Optional[str] = None
    exception_type: Optional[str] = None
    exception_addr: Optional[int] = None
    raw: Optional[str] = None


class ModuleResolver:
    def __init__(
        self,
        modules: Sequence[ModuleRule],
        maps: Sequence[MapEntry],
        addr2line: Addr2lineManager,
    ) -> None:
        self.modules = list(modules)
        self.maps = list(maps)
        self.map_index = MapIndex(maps)
        self.addr2line = addr2line
        self.module_hits = Counter()

    @staticmethod
    def _score_candidate(
        module: ModuleRule,
        mapping: Optional[MapEntry],
        segment: LoadSegment,
    ) -> int:
        score = 100
        if mapping is not None:
            score += 30
            if "x" in mapping.perms:
                score += 30
            if module.map_alias_matches(mapping):
                score += 100
            if permission_compatible(mapping.perms, segment.flags):
                score += 20
        if module.runtime_ranges:
            score += 100
        if module.context_ids is not None:
            score += 50
        if module.els is not None:
            score += 30
        return score

    def _best_candidate(
        self, addr: int, context: ContextState, mapping: Optional[MapEntry]
    ) -> Optional[Tuple[int, ModuleRule, int, LoadSegment]]:
        best: Optional[Tuple[int, ModuleRule, int, LoadSegment]] = None
        for module in self.modules:
            if module.image is None or module.load_bias is None:
                continue
            if not module.context_matches(context):
                continue
            if not module.runtime_range_matches(addr):
                continue

            elf_va = addr - module.load_bias
            segment = module.image.segment_for_va(elf_va, executable_only=True)
            if segment is None:
                continue

            score = self._score_candidate(module, mapping, segment)
            if best is None or score > best[0]:
                best = (score, module, elf_va, segment)
        return best

    def resolve(
        self, addr: int, context: ContextState, *, with_source: bool = True
    ) -> Resolution:
        mapping = self.map_index.find(addr)
        if mapping is not None and "x" not in mapping.perms:
            return Resolution(
                runtime_addr=addr,
                mapping=mapping,
                module=None,
                elf_va=None,
                file_offset=None,
                segment=None,
                section=None,
                symbol=None,
                symbol_offset=None,
                symbol_contains=False,
                source=None,
                reason="address-in-non-executable-map",
            )

        candidate = self._best_candidate(addr, context, mapping)
        if candidate is None:
            reason = "no-configured-elf"
            if mapping is None:
                reason = "address-not-in-maps"
            elif "x" not in mapping.perms:
                reason = "address-in-non-executable-map"
            return Resolution(
                runtime_addr=addr,
                mapping=mapping,
                module=None,
                elf_va=None,
                file_offset=None,
                segment=None,
                section=None,
                symbol=None,
                symbol_offset=None,
                symbol_contains=False,
                source=None,
                reason=reason,
            )

        _, module, elf_va, segment = candidate
        assert module.image is not None

        section = module.image.section_for_va(elf_va)
        symbol, symbol_offset, contains = module.image.symbol_for_va(elf_va)
        file_offset = segment.file_offset_for_va(elf_va)
        source = (
            self.addr2line.query(module.elf_path.resolve(), elf_va)
            if with_source
            else None
        )
        self.module_hits[module.name] += 1

        return Resolution(
            runtime_addr=addr,
            mapping=mapping,
            module=module,
            elf_va=elf_va,
            file_offset=file_offset,
            segment=segment,
            section=section,
            symbol=symbol,
            symbol_offset=symbol_offset,
            symbol_contains=contains,
            source=source,
            reason=None,
        )

    def resolve_fast(self, addr: int, context: ContextState) -> FastResolution:
        mapping = self.map_index.find(addr)
        if mapping is not None and "x" not in mapping.perms:
            return FastResolution(
                runtime_addr=addr,
                mapping=mapping,
                module=None,
                elf_va=None,
                segment=None,
                raw=None,
                reason="address-in-non-executable-map",
            )

        candidate = self._best_candidate(addr, context, mapping)
        if candidate is None:
            reason = "no-configured-elf"
            if mapping is None:
                reason = "address-not-in-maps"
            elif "x" not in mapping.perms:
                reason = "address-in-non-executable-map"
            return FastResolution(
                runtime_addr=addr,
                mapping=mapping,
                module=None,
                elf_va=None,
                segment=None,
                raw=None,
                reason=reason,
            )

        _, module, elf_va, segment = candidate
        assert module.image is not None
        raw = module.image.read_bytes_from_segment(segment, elf_va, 4)
        return FastResolution(
            runtime_addr=addr,
            mapping=mapping,
            module=module,
            elf_va=elf_va,
            segment=segment,
            raw=raw,
            reason=None if raw is not None else "elf-bytes-unavailable",
        )


# ---------------------------------------------------------------------------
# ptm2human_trbe 文本解析
# ---------------------------------------------------------------------------


TRACE_ID_RE = re.compile(r"^Decode trace stream of ID\s+(0x[0-9a-fA-F]+|\d+)(.*)$")
TRACE_INFO_RE = re.compile(r"^TraceInfo\s*-\s*(.*)$")
ADDR_RE = re.compile(
    r"Address - Instruction address\s+(0x[0-9a-fA-F]+),\s*Instruction set\s+(\S+)"
)
CTX_CID_RE = re.compile(r"Context ID\s*=\s*(0x[0-9a-fA-F]+)")
CTX_VMID_RE = re.compile(r"VMID\s*=\s*(0x[0-9a-fA-F]+)")
CTX_EL_RE = re.compile(r"Exception level\s*=\s*(EL\d)")
CTX_SEC_RE = re.compile(r"Security\s*=\s*([^,\s]+)")
TIMESTAMP_RE = re.compile(r"Timestamp\s*-\s*(\d+)")
ATOM_RE = re.compile(r"^ATOM\s*-\s*([EN]+)\s*$")
COMMIT_RE = re.compile(r"^Commit\s*-\s*(\d+)\s*$")
CANCEL_RE = re.compile(r"^Cancel\s*-\s*(\d+)(?:.*)?$")
EXCEPTION_PACKET_RE = re.compile(r"^Exception(?:\s*-\s*(.*))?$")
EXCEPTION_TYPE_RE = re.compile(r"exception type\s+([^,]+)", re.IGNORECASE)
EXCEPTION_ADDR_RE = re.compile(r"\baddress\s+(0x[0-9a-fA-F]+)", re.IGNORECASE)
TRACE_INFO_END_RE = re.compile(r"\bcc_threshold\s*=", re.IGNORECASE)


def parse_trace_info_fields(raw: str) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for part in (item.strip().rstrip(",") for item in raw.split(",")):
        if not part:
            continue
        if "=" not in part:
            continue
        key, value = (item.strip() for item in part.split("=", 1))
        normalized_key = key.lower().replace(" ", "_")
        if normalized_key not in {"p0_key", "curr_spec_depth", "cc_threshold"}:
            continue
        if re.fullmatch(r"0x[0-9a-fA-F]+", value):
            fields[normalized_key] = value.lower()
        else:
            try:
                fields[normalized_key] = int(value, 10)
            except ValueError:
                fields[normalized_key] = value
    return fields


def parse_exception_packet(raw: str) -> Tuple[Optional[str], Optional[int]]:
    detail_match = EXCEPTION_PACKET_RE.match(raw)
    detail = detail_match.group(1) if detail_match else raw
    if not detail:
        return None, None
    type_match = EXCEPTION_TYPE_RE.search(detail)
    addr_match = EXCEPTION_ADDR_RE.search(detail)
    exception_type = type_match.group(1).strip() if type_match else None
    exception_addr = int(addr_match.group(1), 16) if addr_match else None
    return exception_type, exception_addr


class EventWriter:
    """输出事件。

    JSONL 默认采用最小结构：
    - trace_stream/trace_info/context 作为独立元数据事件写入，不重复塞进每行；
    - 不默认写入 discontinuity/raw/commit 等调试事件；
    - timestamp 默认不写入 JSONL；显式开启时只保留数值，并自动去重；
    - 所有地址统一使用 0x 前缀的十六进制字符串；
    - 成功或跳过的 Commit 默认不写入，只保留内部统计；
    - Address 与 instruction_range 只保留恢复和训练所需字段。

    使用 --detailed 时恢复逐指令和 ELF 调试字段。
    discontinuity/commit/raw 等调试类事件由
    --emit-metadata-events 单独控制，避免详细指令模式意外放大 JSONL。
    timestamp 由 --emit-timestamps 单独控制。
    """

    def __init__(
        self,
        stream: TextIO,
        *,
        detailed: bool = False,
        include_module: bool = False,
        emit_metadata_events: bool = False,
        emit_timestamp_events: bool = False,
    ) -> None:
        self.stream = stream
        self.detailed = detailed
        self.include_module = include_module
        self.emit_metadata_events = emit_metadata_events
        self.emit_timestamp_events = emit_timestamp_events
        self.compact_json = not detailed
        self._last_timestamp_emitted: Optional[int] = None

    @staticmethod
    def _hex(value: Optional[int], *, width: int = 0) -> Optional[str]:
        if value is None:
            return None
        if width > 0:
            return f"0x{value:0{width}x}"
        return f"0x{value:x}"

    @classmethod
    def _clean_json(cls, value: Any) -> Any:
        """递归删除 None，避免 source:null、reason:null 等无效字段。"""
        if isinstance(value, dict):
            return {
                key: cls._clean_json(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, list):
            return [cls._clean_json(item) for item in value if item is not None]
        return value

    @staticmethod
    def _context_record(context: ContextState) -> Dict[str, Any]:
        return context.snapshot()

    def write_header(
        self,
        trace_path: Path,
        maps_path: Optional[Path],
        modules: Sequence[ModuleRule],
        page_size: int,
    ) -> None:
        if self.compact_json:
            module_records: List[Dict[str, Any]] = []
            for module in modules:
                item: Dict[str, Any] = {
                    "elf": str(module.elf_path),
                    "load_bias": self._hex(module.load_bias),
                }
                if self.include_module:
                    item["name"] = module.name
                module_records.append(item)
            self.write_json(
                {
                    "event": "header",
                    "version": SCRIPT_VERSION,
                    "trace": str(trace_path),
                    "maps": str(maps_path) if maps_path else None,
                    "page_size": self._hex(page_size),
                    "modules": module_records,
                }
            )
            return

        self.write_json(
            {
                "event": "header",
                "version": SCRIPT_VERSION,
                "trace": str(trace_path),
                "maps": str(maps_path) if maps_path else None,
                "page_size": self._hex(page_size),
                "modules": [
                    {
                        "name": module.name,
                        "elf": str(module.elf_path),
                        "load_bias": self._hex(module.load_bias),
                        "inferred_from": module.inferred_from,
                        "context_ids": [self._hex(v) for v in sorted(module.context_ids)]
                        if module.context_ids
                        else None,
                        "vmids": [self._hex(v) for v in sorted(module.vmids)]
                        if module.vmids
                        else None,
                        "els": sorted(module.els) if module.els else None,
                    }
                    for module in modules
                ],
            }
        )

    def write_json(self, record: Dict[str, Any]) -> None:
        cleaned = self._clean_json(record)
        self.stream.write(
            json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")) + "\n"
        )

    def trace_stream_line(self, context: ContextState, raw: str) -> None:
        self.write_json(
            {
                "event": "trace_stream",
                "trace_id": self._hex(context.trace_id),
                "raw": raw,
            }
        )

    def trace_info_line(self, context: ContextState, fields: Dict[str, Any]) -> None:
        record: Dict[str, Any] = {
            "event": "trace_info",
            "trace_id": self._hex(context.trace_id),
        }
        record.update(fields)
        self.write_json(record)

    def context_line(self, context: ContextState, raw: str) -> None:
        # Context 作为独立元数据事件写入，避免重复放进每条训练事件。
        self.write_json(
            {
                "event": "context",
                "trace_id": self._hex(context.trace_id),
                "context_id": self._hex(context.context_id),
                "vmid": self._hex(context.vmid),
                "el": context.el,
                "security": context.security,
                "isa": context.isa,
            }
        )

    def discontinuity_line(self, context: ContextState, raw: str) -> None:
        # 不保留原始自然语言。只在显式元数据模式下输出结构化原因。
        if not self.emit_metadata_events:
            return
        lower = raw.lower()
        reason = (
            "trace_stream_start"
            if lower.startswith("decode trace stream")
            else "discard"
            if lower.startswith("discard")
            else "conditional_flush"
            if lower.startswith("conditional flush")
            else "trace_on"
            if lower.startswith("traceon")
            else "discontinuity"
        )
        self.write_json({"event": "discontinuity", "reason": reason})

    def trace_control_line(self, context: ContextState, raw: str, kind: str) -> None:
        if not self.emit_metadata_events:
            return
        self.write_json({"event": "trace_control", "kind": kind})

    def timestamp_line(self, context: ContextState) -> None:
        if not self.emit_timestamp_events:
            return
        if context.timestamp is None:
            return
        # 无论是否 detailed，时间戳都只保留数值，且连续重复值去重。
        if context.timestamp == self._last_timestamp_emitted:
            return
        self._last_timestamp_emitted = context.timestamp
        self.write_json({"event": "timestamp", "value": context.timestamp})

    def exception_line(
        self,
        context: ContextState,
        raw: str,
        *,
        exception_type: Optional[str] = None,
        exception_addr: Optional[int] = None,
        status: str = "observed",
        reason: Optional[str] = None,
        target: Optional[Resolution] = None,
    ) -> None:
        lower = raw.lower()
        record: Dict[str, Any] = {
            "event": "exception_return"
            if lower.startswith("exception return")
            else "exception",
            "timestamp": context.timestamp,
            "exception_type": exception_type,
            "address": self._hex(exception_addr, width=16),
            "status": status,
            "reason": reason,
            "raw": raw,
        }
        if target is not None:
            record.update(
                {
                    "target_address": self._hex(target.runtime_addr, width=16),
                    "target_elf_addr": self._hex(target.elf_va),
                    "target_symbol": self._symbol_text(target),
                    "target_map": target.mapping.display_name
                    if target.mapping is not None
                    else None,
                    "target_resolved": target.module is not None,
                }
            )
            if self.include_module and target.module is not None:
                record["target_module"] = target.module.name
        self.write_json(record)

    def address_line(self, context: ContextState, resolution: Resolution) -> None:
        mapping = resolution.mapping
        module = resolution.module
        segment = resolution.segment
        section = resolution.section
        symbol = resolution.symbol
        symbol_text = self._symbol_text(resolution)

        if self.compact_json:
            record: Dict[str, Any] = {
                "event": "address",
                "address": self._hex(resolution.runtime_addr, width=16),
                "reason": resolution.reason,
            }
            if mapping is not None:
                record.update(
                    {
                        "map": mapping.display_name,
                        "map_perms": mapping.perms,
                        "map_offset": self._hex(
                            mapping.offset_of(resolution.runtime_addr)
                        ),
                    }
                )
            if module is not None:
                record.update(
                    {
                        "elf_addr": self._hex(resolution.elf_va),
                        "section": section.name if section is not None else None,
                        "symbol": symbol_text,
                        "source": resolution.source,
                    }
                )
                if self.include_module:
                    record["module"] = module.name
            self.write_json(record)
            return

        record = {
            "event": "address",
            "context": self._context_record(context),
            "runtime_addr": self._hex(resolution.runtime_addr, width=16),
            "reason": resolution.reason,
        }
        if mapping is not None:
            record["mapping"] = {
                "start": self._hex(mapping.start, width=16),
                "end": self._hex(mapping.end, width=16),
                "perms": mapping.perms,
                "pgoff": self._hex(mapping.pgoff),
                "path": mapping.path,
                "offset": self._hex(mapping.offset_of(resolution.runtime_addr)),
            }
        if module is not None:
            if self.include_module:
                record["module"] = {
                    "name": module.name,
                    "elf": str(module.elf_path),
                    "load_bias": self._hex(module.load_bias),
                }
            record["elf_va"] = self._hex(resolution.elf_va)
            record["file_offset"] = self._hex(resolution.file_offset)
        if segment is not None:
            record["segment"] = {
                "index": segment.index,
                "p_offset": self._hex(segment.p_offset),
                "p_vaddr": self._hex(segment.p_vaddr),
                "p_filesz": self._hex(segment.p_filesz),
                "p_memsz": self._hex(segment.p_memsz),
                "flags": segment.flags,
                "align": self._hex(segment.align),
                "perms": segment.perms,
            }
        if section is not None:
            record["section"] = {
                "name": section.name,
                "addr": self._hex(section.addr),
                "size": self._hex(section.size),
                "flags": section.flags,
                "section_type": section.section_type,
            }
        if symbol is not None:
            record["symbol"] = {
                "name": symbol.name,
                "addr": self._hex(symbol.addr),
                "size": self._hex(symbol.size),
                "bind": symbol.bind,
                "symbol_type": symbol.symbol_type,
                "section_index": symbol.section_index,
            }
            record["symbol_offset"] = self._hex(resolution.symbol_offset)
            record["symbol_contains"] = resolution.symbol_contains
        record["source"] = resolution.source
        self.write_json(record)

    @staticmethod
    def _symbol_text(resolution: Resolution) -> Optional[str]:
        if resolution.symbol is None:
            return None
        relation = "+" if resolution.symbol_contains else "~+"
        return f"{resolution.symbol.name}{relation}0x{resolution.symbol_offset or 0:x}"

    def instruction_line(
        self,
        context: ContextState,
        resolution: Resolution,
        instruction: "DecodedInstruction",
        atom: Optional[str] = None,
        taken: Optional[bool] = None,
        next_pc: Optional[int] = None,
    ) -> None:
        """仅详细模式输出单条指令。"""
        symbol_text = self._symbol_text(resolution)

        record: Dict[str, Any] = {
            "event": "instruction",
            "context": self._context_record(context),
            "runtime_addr": self._hex(instruction.runtime_addr, width=16),
            "elf_va": self._hex(resolution.elf_va),
            "file_offset": self._hex(resolution.file_offset),
            "symbol": symbol_text,
            "source": resolution.source,
            "bytes": instruction.raw.hex(),
            "mnemonic": instruction.mnemonic,
            "op_str": instruction.op_str,
            "kind": instruction.kind,
            "target": self._hex(instruction.target, width=16),
            "atom": atom,
            "taken": taken,
            "next_pc": self._hex(next_pc, width=16),
        }
        if self.include_module and resolution.module is not None:
            record["module"] = resolution.module.name
        self.write_json(record)

    def instruction_range_line(
        self,
        context: ContextState,
        start_resolution: Resolution,
        end_resolution: Resolution,
        waypoint: "DecodedInstruction",
        instruction_count: int,
        atom: str,
        taken: Optional[bool],
        next_pc: Optional[int],
        *,
        status: str = "resolved",
        reason: Optional[str] = None,
    ) -> None:
        start_symbol = self._symbol_text(start_resolution)
        end_symbol = self._symbol_text(end_resolution)
        runtime_end = waypoint.runtime_addr
        runtime_end_exclusive = runtime_end + 4
        elf_end = end_resolution.elf_va
        elf_end_exclusive = elf_end + 4 if elf_end is not None else None
        asm_text = (
            waypoint.mnemonic + (" " + waypoint.op_str if waypoint.op_str else "")
        ).strip()

        if self.compact_json:
            record: Dict[str, Any] = {
                "event": "instruction_range",
                "atom": atom,
                "status": status if status != "resolved" else None,
                "reason": reason,
                "runtime_start": self._hex(
                    start_resolution.runtime_addr, width=16
                ),
                "runtime_end": self._hex(runtime_end, width=16),
                "elf_start": self._hex(start_resolution.elf_va),
                "elf_end": self._hex(elf_end),
                "instruction_count": instruction_count,
                "start_symbol": start_symbol,
                "end_symbol": end_symbol,
                "waypoint": {
                    "address": self._hex(waypoint.runtime_addr, width=16),
                    "elf_addr": self._hex(end_resolution.elf_va),
                    "asm": asm_text,
                    "kind": waypoint.kind,
                    "target": self._hex(waypoint.target, width=16),
                    "taken": taken,
                },
                "next_pc": self._hex(next_pc, width=16),
            }
            if self.include_module and start_resolution.module is not None:
                record["module"] = start_resolution.module.name
            self.write_json(record)
            return

        record = {
            "event": "instruction_range",
            "context": self._context_record(context),
            "atom": atom,
            "status": status,
            "reason": reason,
            "runtime_start": self._hex(start_resolution.runtime_addr, width=16),
            "runtime_end": self._hex(runtime_end, width=16),
            "runtime_end_exclusive": self._hex(runtime_end_exclusive, width=16),
            "elf_start": self._hex(start_resolution.elf_va),
            "elf_end": self._hex(elf_end),
            "elf_end_exclusive": self._hex(elf_end_exclusive),
            "instruction_count": instruction_count,
            "start_symbol": start_symbol,
            "end_symbol": end_symbol,
            "waypoint": {
                "runtime_addr": self._hex(waypoint.runtime_addr, width=16),
                "elf_va": self._hex(end_resolution.elf_va),
                "mnemonic": waypoint.mnemonic,
                "op_str": waypoint.op_str,
                "kind": waypoint.kind,
                "target": self._hex(waypoint.target, width=16),
                "taken": taken,
            },
            "next_pc": self._hex(next_pc, width=16),
        }
        if self.include_module and start_resolution.module is not None:
            record["module"] = start_resolution.module.name
        self.write_json(record)

    def flow_gap_line(
        self,
        context: ContextState,
        reason: str,
        pc: Optional[int],
        detail: Optional[str] = None,
    ) -> None:
        if self.compact_json:
            self.write_json(
                {
                    "event": "flow_gap",
                    "reason": reason,
                    "pc": self._hex(pc, width=16),
                    "detail": detail,
                }
            )
            return
        self.write_json(
            {
                "event": "flow_gap",
                "context": self._context_record(context),
                "reason": reason,
                "pc": self._hex(pc, width=16),
                "detail": detail,
            }
        )

    def commit_line(
        self, context: ContextState, count: int, resolved: int, skipped: int
    ) -> None:
        # Commit 属于解码器控制信息，默认只计入统计。
        if not self.emit_metadata_events:
            return
        self.write_json(
            {
                "event": "commit",
                "count": count,
                "resolved": resolved,
                "skipped": skipped,
            }
        )

    def pending_atoms_discarded_line(
        self, context: ContextState, count: int, reason: str
    ) -> None:
        if count <= 0:
            return
        if not self.emit_metadata_events:
            return
        self.write_json(
            {
                "event": "pending_atoms_discarded",
                "count": count,
                "reason": reason,
            }
        )

    def raw_line(self, raw: str) -> None:
        if not self.emit_metadata_events:
            return
        self.write_json({"event": "raw", "value": raw})


# ---------------------------------------------------------------------------
# AArch64 指令流恢复
# ---------------------------------------------------------------------------


def sign_extend(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


@dataclass(frozen=True)
class DecodedInstruction:
    runtime_addr: int
    raw: bytes
    word: int
    mnemonic: str
    op_str: str
    kind: str
    target: Optional[int] = None


class AArch64InstructionDecoder:
    """只依赖编码字段恢复控制流；Capstone 仅增强反汇编文本。"""

    def __init__(self, little_endian: bool = True) -> None:
        self.little_endian = little_endian
        self._cs = None
        if Cs is not None:
            mode = CS_MODE_ARM | (
                CS_MODE_LITTLE_ENDIAN if little_endian else CS_MODE_BIG_ENDIAN
            )
            self._cs = Cs(CS_ARCH_ARM64, mode)
            self._cs.detail = False

    def decode(self, runtime_addr: int, raw: bytes) -> Optional[DecodedInstruction]:
        if len(raw) != 4:
            return None
        byteorder = "little" if self.little_endian else "big"
        word = int.from_bytes(raw, byteorder=byteorder, signed=False)
        mnemonic = f".word 0x{word:08x}"
        op_str = ""
        if self._cs is not None:
            decoded = next(iter(self._cs.disasm(raw, runtime_addr, count=1)), None)
            if decoded is not None:
                mnemonic = decoded.mnemonic
                op_str = decoded.op_str

        kind = "normal"
        target: Optional[int] = None

        # B.cond: imm19 << 2
        if (word & 0xFF000010) == 0x54000000:
            imm19 = (word >> 5) & 0x7FFFF
            target = runtime_addr + (sign_extend(imm19, 19) << 2)
            kind = "conditional_branch"
        # CBZ / CBNZ, 32-bit and 64-bit forms.
        elif (word & 0x7E000000) == 0x34000000:
            imm19 = (word >> 5) & 0x7FFFF
            target = runtime_addr + (sign_extend(imm19, 19) << 2)
            kind = "conditional_branch"
        # TBZ / TBNZ.
        elif (word & 0x7E000000) == 0x36000000:
            imm14 = (word >> 5) & 0x3FFF
            target = runtime_addr + (sign_extend(imm14, 14) << 2)
            kind = "conditional_branch"
        # BL immediate.
        elif (word & 0xFC000000) == 0x94000000:
            imm26 = word & 0x03FFFFFF
            target = runtime_addr + (sign_extend(imm26, 26) << 2)
            kind = "direct_call"
        # B immediate.
        elif (word & 0xFC000000) == 0x14000000:
            imm26 = word & 0x03FFFFFF
            target = runtime_addr + (sign_extend(imm26, 26) << 2)
            kind = "direct_branch"
        # BR / BLR / RET register forms.
        elif (word & 0xFFFFFC1F) == 0xD61F0000:
            kind = "indirect_branch"
        elif (word & 0xFFFFFC1F) == 0xD63F0000:
            kind = "indirect_call"
        elif (word & 0xFFFFFC1F) == 0xD65F0000:
            kind = "return"
        # Pointer Authentication 返回。即使 Capstone 版本较旧，也要识别 RETAA/RETAB。
        elif word in {0xD65F0BFF, 0xD65F0FFF}:
            kind = "return"
        # ERET / DRPS.
        elif word in {0xD69F03E0, 0xD6BF03E0}:
            kind = "exception_return"
        # SVC/HVC/SMC/BRK/HLT 类同步异常。
        elif (word & 0xFFE0001F) in {
            0xD4000001,
            0xD4000002,
            0xD4000003,
            0xD4200000,
            0xD4400000,
        }:
            kind = "exception_instruction"

        # Capstone 可识别较新的 PAC 分支/返回指令，补充内置分类。
        mnemonic_lower = mnemonic.lower()
        if kind == "normal":
            if mnemonic_lower in {"retaa", "retab"}:
                kind = "return"
            elif mnemonic_lower.startswith("blra"):
                kind = "indirect_call"
            elif mnemonic_lower.startswith("bra"):
                kind = "indirect_branch"
            elif mnemonic_lower in {"eret", "eretaa", "eretab", "drps"}:
                kind = "exception_return"

        return DecodedInstruction(
            runtime_addr=runtime_addr,
            raw=raw,
            word=word,
            mnemonic=mnemonic,
            op_str=op_str,
            kind=kind,
            target=target,
        )


class InstructionFollower:
    """
    在两个 Address 同步点之间，使用 ELF 机器码和 ATOM E/N 保守恢复路径。

    重要限制：ptm2human 文本不是无损 OpenCSD 输入。本实现只处理常见 AArch64
    条件分支 P0 元素；任何不能静态确定的情况都停止当前区间，不猜测目标。
    """

    def __init__(
        self,
        resolver: ModuleResolver,
        writer: EventWriter,
        stats: Dict[str, int],
        max_instructions_per_atom: int,
        with_source: bool,
        detailed: bool,
    ) -> None:
        self.resolver = resolver
        self.writer = writer
        self.stats = stats
        self.max_instructions_per_atom = max_instructions_per_atom
        self.with_source = with_source
        self.detailed = detailed
        self.current_pc: Optional[int] = None
        self.context_generation: Optional[int] = None
        self.synced = False
        self.call_stack: List[int] = []
        self._decoders: Dict[Tuple[Path, bool], AArch64InstructionDecoder] = {}
        self.last_gap_reason: Optional[str] = None
        self.pending_target: Optional[PendingAddressTarget] = None

    def _decoder_for(self, module: ModuleRule) -> AArch64InstructionDecoder:
        assert module.image is not None
        key = (module.elf_path.resolve(), module.image.little_endian)
        decoder = self._decoders.get(key)
        if decoder is None:
            decoder = AArch64InstructionDecoder(module.image.little_endian)
            self._decoders[key] = decoder
        return decoder

    def set_anchor(
        self,
        resolution: Resolution,
        context: ContextState,
        *,
        preserve_call_stack: bool = False,
    ) -> bool:
        if not preserve_call_stack:
            self.call_stack.clear()
        self.context_generation = context.generation
        self.last_gap_reason = None
        if (
            resolution.module is None
            or resolution.elf_va is None
            or resolution.file_offset is None
            or resolution.segment is None
            or "x" not in resolution.segment.perms
            or (
                resolution.mapping is not None
                and "x" not in resolution.mapping.perms
            )
        ):
            self.current_pc = None
            self.synced = False
            return False
        self.current_pc = resolution.runtime_addr
        self.synced = True
        self.stats["flow_anchor"] += 1
        return True

    @staticmethod
    def _same_pending_context(
        pending: PendingAddressTarget, context: ContextState
    ) -> bool:
        return (
            pending.context_generation == context.generation
            and pending.trace_id == context.trace_id
            and pending.context_id == context.context_id
            and pending.vmid == context.vmid
            and pending.el == context.el
            and pending.security == context.security
        )

    def defer_address_target(
        self,
        context: ContextState,
        resolution: Resolution,
        instruction: DecodedInstruction,
        *,
        range_start_resolution: Optional[Resolution] = None,
        instruction_count: int = 0,
        atom: Optional[str] = None,
        taken: Optional[bool] = None,
        reason: Optional[str] = None,
    ) -> None:
        """保存需要 Address 包补全目标的动态 waypoint。"""
        if self.pending_target is not None:
            self.invalidate_pending_target(
                context,
                "new-pending-target-before-previous-resolved",
                detail=instruction.mnemonic,
            )

        return_address: Optional[int] = None
        if instruction.kind == "indirect_call":
            return_address = instruction.runtime_addr + 4
            self.call_stack.append(return_address)
            if len(self.call_stack) > 4096:
                self.call_stack = self.call_stack[-4096:]
                self.stats["return_stack_truncated"] += 1

        self.pending_target = PendingAddressTarget(
            source_pc=instruction.runtime_addr,
            source_elf_va=resolution.elf_va,
            source_symbol=self.writer._symbol_text(resolution),
            kind=instruction.kind,
            mnemonic=(
                instruction.mnemonic
                + (" " + instruction.op_str if instruction.op_str else "")
            ).strip(),
            context_generation=context.generation,
            trace_id=context.trace_id,
            context_id=context.context_id,
            vmid=context.vmid,
            el=context.el,
            security=context.security,
            timestamp=context.timestamp,
            return_address=return_address,
            range_context=ContextState(**context.snapshot()),
            range_start_resolution=range_start_resolution or resolution,
            range_end_resolution=resolution,
            range_waypoint=instruction,
            range_instruction_count=instruction_count,
            range_atom=atom,
            range_taken=taken,
            range_reason=reason,
        )
        self.current_pc = None
        self.synced = False
        self.context_generation = context.generation
        self.last_gap_reason = "waiting-for-address-target"
        self.stats["pending_address_target"] += 1

    def defer_exception_target(
        self,
        context: ContextState,
        raw: str,
        *,
        exception_type: Optional[str],
        exception_addr: Optional[int],
    ) -> None:
        """保存需要下一条 Address 包补全目标的异常包。"""
        if self.pending_target is not None:
            self.invalidate_pending_target(
                context,
                "new-pending-target-before-previous-resolved",
                detail=raw,
            )

        source_pc = (
            exception_addr
            if exception_addr is not None
            else self.current_pc
            if self.current_pc is not None
            else 0
        )
        self.pending_target = PendingAddressTarget(
            source_pc=source_pc,
            source_elf_va=None,
            source_symbol=None,
            kind="exception",
            mnemonic=raw,
            context_generation=context.generation,
            trace_id=context.trace_id,
            context_id=context.context_id,
            vmid=context.vmid,
            el=context.el,
            security=context.security,
            timestamp=context.timestamp,
            range_context=ContextState(**context.snapshot()),
            range_reason="exception-target-awaiting-address-packet",
            exception_type=exception_type,
            exception_addr=exception_addr,
            raw=raw,
        )
        self.current_pc = None
        self.synced = False
        self.context_generation = context.generation
        self.last_gap_reason = "waiting-for-exception-target"
        self.stats["pending_exception_target"] += 1

    def _emit_pending_range(
        self,
        pending: PendingAddressTarget,
        context: ContextState,
        target: Optional[Resolution],
        *,
        status: str,
        reason: Optional[str],
    ) -> None:
        if (
            pending.range_start_resolution is None
            or pending.range_end_resolution is None
            or pending.range_waypoint is None
            or pending.range_atom is None
        ):
            return

        self.writer.instruction_range_line(
            context=pending.range_context or context,
            start_resolution=pending.range_start_resolution,
            end_resolution=pending.range_end_resolution,
            waypoint=pending.range_waypoint,
            instruction_count=pending.range_instruction_count,
            atom=pending.range_atom,
            taken=pending.range_taken,
            next_pc=target.runtime_addr if target is not None else None,
            status=status,
            reason=reason,
        )
        self.stats["instruction_range"] += 1

    def resolve_pending_target(
        self, resolution: Resolution, context: ContextState
    ) -> bool:
        """若当前 Address 紧随动态 waypoint，则输出动态边并返回是否保留调用栈。"""
        pending = self.pending_target
        if pending is None:
            return False
        if not self._same_pending_context(pending, context):
            self.invalidate_pending_target(
                context,
                "pending-target-context-mismatch",
                detail=f"target=0x{resolution.runtime_addr:x}",
            )
            return False
        if pending.kind == "exception":
            target_resolved = resolution.module is not None
            self.writer.exception_line(
                pending.range_context or context,
                pending.raw or pending.mnemonic,
                exception_type=pending.exception_type,
                exception_addr=pending.exception_addr,
                status="target_resolved"
                if target_resolved
                else "target_address_unresolved",
                reason=pending.range_reason
                if target_resolved
                else "; ".join(
                    item
                    for item in (pending.range_reason, resolution.reason)
                    if item
                ),
                target=resolution,
            )
            self.stats["exception_target_resolved_by_address"] += 1
            self.pending_target = None
            self.last_gap_reason = None
            return False
        if (
            resolution.module is None
            or resolution.elf_va is None
            or resolution.segment is None
            or (
                resolution.mapping is not None
                and "x" not in resolution.mapping.perms
            )
        ):
            self.invalidate_pending_target(
                context,
                "pending-target-unresolved",
                detail=resolution.reason or f"target=0x{resolution.runtime_addr:x}",
            )
            return False

        self._emit_pending_range(
            pending,
            context,
            resolution,
            status="target_resolved",
            reason=pending.range_reason,
        )
        if pending.kind == "return":
            self.stats["return_target_resolved_by_address"] += 1
        else:
            self.stats["indirect_target_resolved_by_address"] += 1
        preserve_call_stack = pending.kind == "indirect_call"
        self.pending_target = None
        self.last_gap_reason = None
        return preserve_call_stack

    def invalidate_pending_target(
        self,
        context: ContextState,
        reason: str,
        *,
        detail: Optional[str] = None,
    ) -> None:
        pending = self.pending_target
        if pending is None:
            return
        if pending.kind == "exception":
            self.writer.exception_line(
                pending.range_context or context,
                pending.raw or pending.mnemonic,
                exception_type=pending.exception_type,
                exception_addr=pending.exception_addr,
                status="target_unresolved",
                reason=reason,
            )
            self.stats["exception_target_unresolved"] += 1
            self.pending_target = None
            self.last_gap_reason = reason
            return
        combined_detail = pending.mnemonic
        if detail:
            combined_detail = f"{combined_detail}; {detail}"
        range_reason = pending.range_reason or reason
        if pending.range_reason and pending.range_reason != reason:
            range_reason = f"{pending.range_reason}; {reason}"
        self._emit_pending_range(
            pending,
            context,
            None,
            status="target_unresolved",
            reason=range_reason,
        )
        self.stats["indirect_target_unresolved"] += 1
        self.stats[f"flow_gap_{reason}"] += 1
        self.writer.flow_gap_line(
            context,
            reason,
            pending.source_pc,
            combined_detail,
        )
        if pending.kind == "indirect_call" and self.call_stack:
            expected = pending.return_address
            if expected is not None and self.call_stack[-1] == expected:
                self.call_stack.pop()
        self.pending_target = None
        self.last_gap_reason = reason

    def finish(self, context: ContextState) -> None:
        self.invalidate_pending_target(context, "end-of-trace-before-address-target")

    def reset(
        self,
        context: ContextState,
        reason: str,
        *,
        emit_gap: bool = False,
        detail: Optional[str] = None,
    ) -> None:
        # Context/异常/断流会使“下一个 Address 是间接目标”的假设失效。
        self.invalidate_pending_target(context, reason, detail=detail)
        old_pc = self.current_pc
        was_synced = self.synced
        self.current_pc = None
        self.synced = False
        self.context_generation = None
        self.call_stack.clear()
        self.last_gap_reason = reason
        if emit_gap and was_synced:
            self.writer.flow_gap_line(context, reason, old_pc, detail)

    def _fail(self, context: ContextState, reason: str, detail: Optional[str] = None) -> bool:
        self.stats[f"flow_gap_{reason}"] += 1
        self.writer.flow_gap_line(context, reason, self.current_pc, detail)
        self.reset(context, reason, emit_gap=False)
        return False

    def _consume_then_gap(
        self,
        context: ContextState,
        reason: str,
        detail: Optional[str] = None,
    ) -> bool:
        """当前 ATOM 已经成功绑定到 waypoint，但后续目标不可恢复。"""
        self.stats["atom_resolved"] += 1
        self.stats[f"flow_gap_{reason}"] += 1
        old_pc = self.current_pc
        self.writer.flow_gap_line(context, reason, old_pc, detail)
        self.reset(context, reason, emit_gap=False)
        return True

    def consume_atom(self, atom: str, context: ContextState) -> bool:
        """
        消费一个 ETMv4/ETE Atom，并输出一个紧凑指令范围。

        默认模式只输出 Address 锚点和 instruction_range；--detailed 时额外输出
        范围内的每条 instruction 记录。
        """
        if atom not in {"E", "N"}:
            return self._fail(context, "unsupported-atom", atom)
        if self.pending_target is not None:
            self.invalidate_pending_target(
                context,
                "atom-before-address-target",
                detail=f"atom={atom}",
            )
            self.stats["atom_skipped_no_anchor"] += 1
            return False
        if not self.synced or self.current_pc is None:
            self.stats["atom_skipped_no_anchor"] += 1
            return False
        if self.context_generation != context.generation:
            return self._fail(context, "context-generation-changed")

        range_start_pc = self.current_pc
        range_start_resolution: Optional[Resolution] = None
        instruction_count = 0

        def emit_range(
            end_resolution: Resolution,
            waypoint: DecodedInstruction,
            *,
            taken: Optional[bool],
            next_pc: Optional[int],
            status: str = "resolved",
            reason: Optional[str] = None,
        ) -> None:
            nonlocal range_start_resolution
            if range_start_resolution is None:
                range_start_resolution = end_resolution
            self.writer.instruction_range_line(
                context=context,
                start_resolution=range_start_resolution,
                end_resolution=end_resolution,
                waypoint=waypoint,
                instruction_count=instruction_count,
                atom=atom,
                taken=taken,
                next_pc=next_pc,
                status=status,
                reason=reason,
            )
            self.stats["instruction_range"] += 1

        for _ in range(self.max_instructions_per_atom):
            assert self.current_pc is not None
            current = self.current_pc
            resolution: Optional[Resolution] = None
            if self.detailed:
                resolution = self.resolver.resolve(
                    current, context, with_source=self.with_source
                )
                if resolution.module is None or resolution.elf_va is None:
                    return self._fail(
                        context,
                        "pc-unresolved",
                        resolution.reason or "unknown",
                    )
                if resolution.mapping is None:
                    return self._fail(context, "pc-not-in-maps")
                if "x" not in resolution.mapping.perms:
                    return self._fail(context, "pc-in-non-executable-map")
                if range_start_resolution is None:
                    range_start_resolution = resolution
                module = resolution.module
                assert module.image is not None
                raw = module.image.read_bytes_for_va(resolution.elf_va, 4)
            else:
                fast = self.resolver.resolve_fast(current, context)
                if fast.module is None or fast.elf_va is None:
                    return self._fail(
                        context,
                        "pc-unresolved",
                        fast.reason or "unknown",
                    )
                if fast.mapping is None:
                    return self._fail(context, "pc-not-in-maps")
                if "x" not in fast.mapping.perms:
                    return self._fail(context, "pc-in-non-executable-map")
                if fast.raw is None:
                    return self._fail(
                        context,
                        fast.reason or "elf-bytes-unavailable",
                    )
                if range_start_resolution is None:
                    range_start_resolution = self.resolver.resolve(
                        current, context, with_source=self.with_source
                    )
                    if (
                        range_start_resolution.module is None
                        or range_start_resolution.elf_va is None
                    ):
                        return self._fail(
                            context,
                            "pc-unresolved",
                            range_start_resolution.reason or "unknown",
                        )
                module = fast.module
                raw = fast.raw

            if raw is None:
                return self._fail(context, "elf-bytes-unavailable")

            instruction = self._decoder_for(module).decode(current, raw)
            if instruction is None:
                return self._fail(context, "instruction-decode-failed")

            instruction_count += 1
            self.stats["instruction"] += 1
            next_pc: Optional[int] = None

            if instruction.kind != "normal" and resolution is None:
                resolution = self.resolver.resolve(
                    current, context, with_source=self.with_source
                )
                if resolution.module is None or resolution.elf_va is None:
                    return self._fail(
                        context,
                        "pc-unresolved",
                        resolution.reason or "unknown",
                    )

            # 条件 waypoint：E 表示采用分支，N 表示不采用。
            if instruction.kind == "conditional_branch":
                if instruction.target is None:
                    return self._fail(context, "conditional-target-missing")
                taken = atom == "E"
                next_pc = instruction.target if taken else current + 4
                if self.detailed:
                    self.writer.instruction_line(
                        context,
                        resolution,
                        instruction,
                        atom=atom,
                        taken=taken,
                        next_pc=next_pc,
                    )
                emit_range(
                    resolution,
                    instruction,
                    taken=taken,
                    next_pc=next_pc,
                )
                self.current_pc = next_pc
                self.stats["atom_resolved"] += 1
                return True

            # B/BL 也是 waypoint，会消费一个 Atom；无条件分支只能对应 E。
            if instruction.kind in {"direct_branch", "direct_call"}:
                if atom != "E":
                    if self.detailed:
                        self.writer.instruction_line(
                            context,
                            resolution,
                            instruction,
                            atom=atom,
                            taken=False,
                        )
                    emit_range(
                        resolution,
                        instruction,
                        taken=False,
                        next_pc=None,
                        status="conflict",
                        reason="n-atom-on-unconditional-direct-branch",
                    )
                    return self._fail(
                        context,
                        "n-atom-on-unconditional-direct-branch",
                        instruction.mnemonic,
                    )
                if instruction.target is None:
                    return self._fail(context, "direct-target-missing")
                next_pc = instruction.target
                if instruction.kind == "direct_call":
                    self.call_stack.append(current + 4)
                    if len(self.call_stack) > 4096:
                        return self._fail(context, "return-stack-overflow")
                if self.detailed:
                    self.writer.instruction_line(
                        context,
                        resolution,
                        instruction,
                        atom=atom,
                        taken=True,
                        next_pc=next_pc,
                    )
                emit_range(
                    resolution,
                    instruction,
                    taken=True,
                    next_pc=next_pc,
                )
                self.current_pc = next_pc
                self.stats["atom_resolved"] += 1
                return True

            # RET/RETAB。若本地调用栈可用则继续，否则等待下一个 Address。
            if instruction.kind == "return":
                if atom != "E":
                    if self.detailed:
                        self.writer.instruction_line(
                            context,
                            resolution,
                            instruction,
                            atom=atom,
                            taken=False,
                        )
                    emit_range(
                        resolution,
                        instruction,
                        taken=False,
                        next_pc=None,
                        status="conflict",
                        reason="n-atom-on-unconditional-return",
                    )
                    return self._fail(
                        context,
                        "n-atom-on-unconditional-return",
                        instruction.mnemonic,
                    )
                if self.call_stack:
                    next_pc = self.call_stack.pop()
                    if self.detailed:
                        self.writer.instruction_line(
                            context,
                            resolution,
                            instruction,
                            atom=atom,
                            taken=True,
                            next_pc=next_pc,
                        )
                    emit_range(
                        resolution,
                        instruction,
                        taken=True,
                        next_pc=next_pc,
                    )
                    self.current_pc = next_pc
                    self.stats["atom_resolved"] += 1
                    return True
                if self.detailed:
                    self.writer.instruction_line(
                        context,
                        resolution,
                        instruction,
                        atom=atom,
                        taken=True,
                    )
                self.defer_address_target(
                    context,
                    resolution,
                    instruction,
                    range_start_resolution=range_start_resolution,
                    instruction_count=instruction_count,
                    atom=atom,
                    taken=True,
                    reason="return-target-awaiting-address-packet",
                )
                self.stats["atom_resolved"] += 1
                return True

            # BR/BLR 等无条件间接 waypoint：目标需要 Address 包。
            if instruction.kind in {"indirect_branch", "indirect_call"}:
                if atom != "E":
                    if self.detailed:
                        self.writer.instruction_line(
                            context,
                            resolution,
                            instruction,
                            atom=atom,
                            taken=False,
                        )
                    emit_range(
                        resolution,
                        instruction,
                        taken=False,
                        next_pc=None,
                        status="conflict",
                        reason="n-atom-on-unconditional-indirect-branch",
                    )
                    return self._fail(
                        context,
                        "n-atom-on-unconditional-indirect-branch",
                        instruction.mnemonic,
                    )
                if self.detailed:
                    self.writer.instruction_line(
                        context,
                        resolution,
                        instruction,
                        atom=atom,
                        taken=True,
                    )
                self.defer_address_target(
                    context,
                    resolution,
                    instruction,
                    range_start_resolution=range_start_resolution,
                    instruction_count=instruction_count,
                    atom=atom,
                    taken=True,
                    reason="indirect-target-awaiting-address-packet",
                )
                self.stats["atom_resolved"] += 1
                return True

            if instruction.kind in {"exception_return", "exception_instruction"}:
                if atom != "E":
                    emit_range(
                        resolution,
                        instruction,
                        taken=False,
                        next_pc=None,
                        status="conflict",
                        reason="n-atom-on-exception-waypoint",
                    )
                    return self._fail(
                        context,
                        "n-atom-on-exception-waypoint",
                        instruction.mnemonic,
                    )
                if self.detailed:
                    self.writer.instruction_line(
                        context,
                        resolution,
                        instruction,
                        atom=atom,
                        taken=True,
                    )
                emit_range(
                    resolution,
                    instruction,
                    taken=True,
                    next_pc=None,
                    status="target_unknown",
                    reason=instruction.kind,
                )
                return self._consume_then_gap(
                    context,
                    instruction.kind,
                    instruction.mnemonic,
                )

            # 普通指令不是 waypoint，不消费当前 Atom，继续寻找下一条 waypoint。
            next_pc = current + 4
            if self.detailed:
                self.writer.instruction_line(
                    context,
                    resolution,
                    instruction,
                    next_pc=next_pc,
                )
            self.current_pc = next_pc

        # 超过扫描上限时，没有可靠 waypoint；不输出伪造的完整范围。
        self.current_pc = range_start_pc
        return self._fail(
            context,
            "max-instructions-before-waypoint",
            f"limit={self.max_instructions_per_atom}",
        )


# ---------------------------------------------------------------------------
# 主处理流程
# ---------------------------------------------------------------------------


def prepare_modules(
    modules: Sequence[ModuleRule],
    maps: Sequence[MapEntry],
    page_size: int,
) -> None:
    for module in modules:
        module.image = ElfImage(module.elf_path, module.name)
        if module.load_bias is None:
            module.load_bias, module.inferred_from = infer_load_bias(module, maps, page_size)
            print(
                f"[INFO] 模块 {module.name} 自动推断 load_bias=0x{module.load_bias:x}: "
                f"{module.inferred_from}",
                file=sys.stderr,
            )
        else:
            module.inferred_from = "explicit"
            print(
                f"[INFO] 模块 {module.name} 使用 load_bias=0x{module.load_bias:x}",
                file=sys.stderr,
            )


def process_trace(
    trace_path: Path,
    maps_path: Optional[Path],
    output_path: Optional[Path],
    modules: Sequence[ModuleRule],
    page_size: int,
    addr2line_path: Optional[str],
    address_source: bool,
    verbose: bool,
    progress_bytes: int,
    only_context_ids: Optional[Set[int]],
    only_els: Optional[Set[str]],
    atom_mode: str,
    max_instructions_per_atom: int,
    instruction_source: bool,
    detailed: bool,
    include_module: bool,
    emit_metadata_events: bool,
    emit_timestamp_events: bool,
    emit_summary: bool = True,
) -> Dict[str, Any]:
    stopwatch = Stopwatch()
    stats: Dict[str, int] = defaultdict(int)
    maps = parse_maps(str(maps_path) if maps_path else None)
    prepare_modules(modules, maps, page_size)

    addr2line = Addr2lineManager(addr2line_path, stopwatch)
    resolver = ModuleResolver(modules, maps, addr2line)

    trace_size = trace_path.stat().st_size
    output_stream: TextIO
    close_output = False
    if output_path:
        output_stream = output_path.open("w", encoding="utf-8")
        close_output = True
    else:
        output_stream = sys.stdout

    effective_include_module = include_module or len(modules) > 1
    writer = EventWriter(
        output_stream,
        detailed=detailed,
        include_module=effective_include_module,
        emit_metadata_events=emit_metadata_events,
        emit_timestamp_events=emit_timestamp_events,
    )
    writer.write_header(trace_path, maps_path, modules, page_size)

    context = ContextState()
    pending_context_change = False
    pending_atoms: Deque[str] = deque()
    trace_info_parts: Optional[List[str]] = None
    pending_resync_reasons: List[str] = []
    pending_resync_pc: Optional[int] = None
    pending_resync_was_synced = False
    read_bytes = 0
    last_report = 0
    follower = InstructionFollower(
        resolver=resolver,
        writer=writer,
        stats=stats,
        max_instructions_per_atom=max_instructions_per_atom,
        with_source=instruction_source,
        detailed=detailed,
    )

    def discard_pending(reason: str) -> None:
        count = len(pending_atoms)
        if count:
            stats["atom_discarded"] += count
            writer.pending_atoms_discarded_line(context, count, reason)
            pending_atoms.clear()

    def reset_flow(reason: str, *, emit_gap: bool = False) -> None:
        discard_pending(reason)
        follower.reset(context, reason, emit_gap=emit_gap)

    def begin_resync(reason: str) -> None:
        nonlocal pending_resync_pc, pending_resync_was_synced
        if not pending_resync_reasons:
            pending_resync_pc = follower.current_pc
            pending_resync_was_synced = follower.synced
            context.generation += 1
            discard_pending("trace-resync")
            follower.reset(context, "trace-resync", emit_gap=False)
        if reason not in pending_resync_reasons:
            pending_resync_reasons.append(reason)

    def finish_resync() -> None:
        nonlocal pending_resync_pc, pending_resync_was_synced
        if not pending_resync_reasons:
            return
        detail = ",".join(pending_resync_reasons)
        if pending_resync_was_synced:
            stats["flow_gap_trace-resync"] += 1
            writer.flow_gap_line(context, "trace-resync", pending_resync_pc, detail)
        stats["trace_resync"] += 1
        pending_resync_reasons.clear()
        pending_resync_pc = None
        pending_resync_was_synced = False

    def is_resync_packet(line_value: str) -> bool:
        return (
            line_value.startswith("Discard")
            or line_value.startswith("Conditional flush")
            or line_value.startswith("TraceInfo")
            or line_value.startswith("TraceOn")
        )

    def emit_trace_info_block() -> None:
        nonlocal trace_info_parts
        if trace_info_parts is None:
            return
        trace_info_fields = parse_trace_info_fields(" ".join(trace_info_parts))
        depth_value = trace_info_fields.get("curr_spec_depth")
        depth = depth_value if isinstance(depth_value, int) else 0
        stats["traceinfo_spec_depth"] = depth
        if depth != 0:
            stats["nonzero_initial_spec_depth"] += 1
            writer.flow_gap_line(
                context,
                "nonzero-initial-spec-depth",
                None,
                f"curr_spec_depth={depth}",
            )
        writer.trace_info_line(context, trace_info_fields)
        trace_info_parts = None

    stopwatch.start("total")
    try:
        with trace_path.open("r", encoding="utf-8", errors="replace") as trace_stream:
            for raw_line in trace_stream:
                stats["lines"] += 1
                read_bytes += len(raw_line.encode("utf-8", errors="replace"))
                line = raw_line.strip()
                if not line:
                    continue

                if progress_bytes > 0 and read_bytes - last_report >= progress_bytes:
                    percent = 100.0 * read_bytes / trace_size if trace_size else 0.0
                    print(
                        f"[PROGRESS] {read_bytes}/{trace_size} ({percent:.1f}%) "
                        f"addr={stats['address']} instruction={stats['instruction']} "
                        f"atom_resolved={stats['atom_resolved']} unresolved={stats['unresolved']}",
                        file=sys.stderr,
                    )
                    last_report = read_bytes

                if trace_info_parts is not None:
                    trace_info_parts.append(line)
                    if TRACE_INFO_END_RE.search(line):
                        emit_trace_info_block()
                    continue

                if pending_resync_reasons and not is_resync_packet(line):
                    finish_resync()

                trace_id_match = (
                    TRACE_ID_RE.search(line)
                    if line.startswith("Decode trace stream")
                    else None
                )
                if trace_id_match:
                    context.trace_id = parse_trace_id(trace_id_match.group(1))
                    context.reset_execution_context()
                    context.generation += 1
                    pending_context_change = False
                    reset_flow("trace-stream-change")
                    stats["trace_stream"] += 1
                    writer.trace_stream_line(context, line)
                    continue

                trace_info_match = (
                    TRACE_INFO_RE.match(line)
                    if line.startswith("TraceInfo")
                    else None
                )
                if trace_info_match:
                    trace_info_parts = [trace_info_match.group(1).strip()]
                    if TRACE_INFO_END_RE.search(trace_info_match.group(1)):
                        emit_trace_info_block()
                    continue

                context_field_seen = False
                context_changed = False

                match = CTX_CID_RE.search(line) if "Context ID" in line else None
                if match:
                    context_field_seen = True
                    value = int(match.group(1), 16)
                    if value != context.context_id:
                        context.context_id = value
                        context_changed = True

                match = CTX_VMID_RE.search(line) if "VMID" in line else None
                if match:
                    context_field_seen = True
                    value = int(match.group(1), 16)
                    if value != context.vmid:
                        context.vmid = value
                        context_changed = True

                match = (
                    CTX_EL_RE.search(line)
                    if "Exception level" in line
                    else None
                )
                if match:
                    context_field_seen = True
                    value = match.group(1).upper()
                    if value != context.el:
                        context.el = value
                        context_changed = True

                match = CTX_SEC_RE.search(line) if "Security" in line else None
                if match:
                    context_field_seen = True
                    value = match.group(1)
                    if value != context.security:
                        context.security = value
                        context_changed = True

                context_packet_finished = False
                if "64-bit instruction" in line:
                    context_field_seen = True
                    context_packet_finished = True
                    if context.isa != "AArch64":
                        context.isa = "AArch64"
                        context_changed = True
                elif "32-bit instruction" in line:
                    context_field_seen = True
                    context_packet_finished = True
                    if context.isa != "AArch32":
                        context.isa = "AArch32"
                        context_changed = True

                if context_changed:
                    pending_context_change = True

                if context_field_seen:
                    if context_packet_finished:
                        if pending_context_change:
                            context.generation += 1
                            reset_flow("context-change")
                            stats["context_change"] += 1
                        writer.context_line(context, line)
                        pending_context_change = False
                    continue

                if pending_context_change:
                    context.generation += 1
                    reset_flow("context-change")
                    stats["context_change"] += 1
                    writer.context_line(context, "context packet completed implicitly")
                    pending_context_change = False

                timestamp_match = (
                    TIMESTAMP_RE.search(line)
                    if line.startswith("Timestamp")
                    else None
                )
                if timestamp_match:
                    context.timestamp = int(timestamp_match.group(1), 10)
                    stats["timestamp"] += 1
                    writer.timestamp_line(context)
                    continue

                if line.startswith("TraceOn"):
                    begin_resync("trace-on")
                    stats["discontinuity"] += 1
                    writer.discontinuity_line(context, line)
                    finish_resync()
                    continue

                if line.startswith("Discard"):
                    begin_resync("discard")
                    stats["discard"] += 1
                    writer.discontinuity_line(context, line)
                    continue

                if line.startswith("Conditional flush"):
                    stats["conditional_flush"] += 1
                    if pending_resync_reasons:
                        if "conditional-flush" not in pending_resync_reasons:
                            pending_resync_reasons.append("conditional-flush")
                    else:
                        writer.trace_control_line(context, line, "conditional_flush")
                    continue

                if line == "Exception" or line.startswith("Exception -"):
                    exception_type, exception_addr = parse_exception_packet(line)
                    context.in_exception = True
                    context.generation += 1
                    discard_pending("exception-boundary")
                    follower.defer_exception_target(
                        context,
                        line,
                        exception_type=exception_type,
                        exception_addr=exception_addr,
                    )
                    stats["exception"] += 1
                    continue

                if (
                    line.startswith("Exception return")
                    or line.startswith("Exception Return")
                    or line.startswith("exception return")
                    or line.startswith("EXCEPTION RETURN")
                ):
                    context.in_exception = False
                    context.generation += 1
                    discard_pending("exception-return")
                    follower.reset(context, "exception-return", emit_gap=False)
                    stats["exception"] += 1
                    writer.exception_line(context, line, status="observed")
                    continue

                cancel_match = CANCEL_RE.match(line) if line.startswith("Cancel") else None
                if (
                    cancel_match
                    or "mispredict" in line
                    or "Mispredict" in line
                    or "MISPREDICT" in line
                ):
                    count = int(cancel_match.group(1)) if cancel_match else 0
                    stats["cancel"] += count
                    reset_flow("cancel-or-mispredict", emit_gap=True)
                    if verbose:
                        writer.raw_line(line)
                    continue

                address_match = (
                    ADDR_RE.search(line)
                    if line.startswith("Address - Instruction address")
                    else None
                )
                if address_match:
                    discard_pending("new-address-packet")
                    isa_text = address_match.group(2)
                    context.isa = (
                        "AArch64"
                        if "aarch64" in isa_text.lower()
                        else "AArch32"
                        if "aarch32" in isa_text.lower()
                        else isa_text
                    )
                    addr = int(address_match.group(1), 16)
                    stats["address"] += 1

                    context_allowed = True
                    if only_context_ids is not None and context.context_id not in only_context_ids:
                        context_allowed = False
                    if only_els is not None and context.el not in only_els:
                        context_allowed = False

                    if context_allowed:
                        stopwatch.start("resolve")
                        resolution = resolver.resolve(
                            addr, context, with_source=address_source
                        )
                        stopwatch.stop("resolve")
                        if resolution.module is None:
                            stats["unresolved"] += 1
                        else:
                            stats["resolved"] += 1
                        if context.isa != "AArch64":
                            stats["unsupported_isa"] += 1
                            follower.reset(context, "unsupported-isa")
                            writer.address_line(context, resolution)
                            writer.flow_gap_line(
                                context,
                                "unsupported-isa",
                                addr,
                                context.isa or "unknown",
                            )
                        else:
                            preserve_call_stack = follower.resolve_pending_target(
                                resolution, context
                            )
                            writer.address_line(context, resolution)
                            if not follower.set_anchor(
                                resolution,
                                context,
                                preserve_call_stack=preserve_call_stack,
                            ):
                                stats["invalid_anchor"] += 1
                    else:
                        stats["filtered_address"] += 1
                        mapping = resolver.map_index.find(addr)
                        resolution = Resolution(
                            runtime_addr=addr,
                            mapping=mapping,
                            module=None,
                            elf_va=None,
                            file_offset=None,
                            segment=None,
                            section=None,
                            symbol=None,
                            symbol_offset=None,
                            symbol_contains=False,
                            source=None,
                            reason="filtered-context",
                        )
                        if verbose:
                            writer.address_line(context, resolution)
                        follower.reset(context, "filtered-address")
                    continue

                atom_match = ATOM_RE.match(line) if line.startswith("ATOM") else None
                if atom_match:
                    atom_string = atom_match.group(1)
                    for atom in atom_string:
                        stats["atom"] += 1
                        if atom_mode == "immediate":
                            if not follower.consume_atom(atom, context):
                                stats["atom_skipped"] += 1
                        else:
                            pending_atoms.append(atom)
                    continue

                commit_match = COMMIT_RE.match(line) if line.startswith("Commit") else None
                if commit_match:
                    count = int(commit_match.group(1), 10)
                    stats["commit"] += count
                    resolved_count = 0
                    skipped_count = 0
                    if atom_mode == "immediate":
                        # ATOM 已在到达时处理；Commit 仅保留为一致性信息。
                        resolved_count = count
                    else:
                        for _ in range(count):
                            if not pending_atoms:
                                skipped_count += 1
                                stats["commit_without_atom"] += 1
                                follower.reset(context, "commit-without-pending-atom")
                                continue
                            atom = pending_atoms.popleft()
                            if follower.consume_atom(atom, context):
                                resolved_count += 1
                            else:
                                skipped_count += 1
                                stats["atom_skipped"] += 1
                    writer.commit_line(context, count, resolved_count, skipped_count)
                    continue

                if verbose:
                    writer.raw_line(line)

        emit_trace_info_block()
        finish_resync()
        discard_pending("end-of-trace")
        follower.finish(context)
    finally:
        stopwatch.stop("total")
        addr2line.close()
        if close_output:
            output_stream.close()

    result: Dict[str, Any] = {
        "trace": str(trace_path),
        "output": str(output_path) if output_path else None,
        "stats": dict(stats),
        "addr2line_cache_hit": addr2line.cache_hits,
        "addr2line_cache_miss": addr2line.cache_misses,
        "module_hits": dict(resolver.module_hits),
        "timing": stopwatch.summary(),
    }

    if emit_summary:
        print("\n[SUMMARY]", file=sys.stderr)
        print(f"  trace: {trace_path}", file=sys.stderr)
        for key in sorted(stats):
            print(f"  {key}: {stats[key]}", file=sys.stderr)
        print(
            f"  addr2line cache: hit={addr2line.cache_hits} miss={addr2line.cache_misses}",
            file=sys.stderr,
        )
        if resolver.module_hits:
            print(
                "  module hits: "
                + ", ".join(f"{name}={count}" for name, count in resolver.module_hits.most_common()),
                file=sys.stderr,
            )
        print(
            "  timing(s): "
            + json.dumps({key: round(value, 6) for key, value in stopwatch.summary().items()}),
            file=sys.stderr,
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 maps + ELF + ATOM 保守恢复 ptm2human_trbe 的 AArch64 指令流。"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--trace",
        help="处理单个 ptm2human_trbe 文本 Trace 文件",
    )
    input_group.add_argument(
        "--trace-folder", "--trace_folder",
        dest="trace_folder",
        help="批量处理目录中的 Trace 文件；不能与 --trace 同时使用",
    )
    parser.add_argument("--maps", help="与 Trace 同一次运行导出的 sysmgr maps")
    parser.add_argument(
        "--out",
        help="单文件模式为输出文件；批量模式为输出目录，缺省自动创建 <trace_folder>_symbolized",
    )

    module_group = parser.add_argument_group("单模块快速模式")
    module_group.add_argument("--elf", help="sysmgr.unstripped.elf")
    module_group.add_argument(
        "--load-bias",
        help="运行时 Load Bias；不传则通过 maps + PT_LOAD 自动推断",
    )
    module_group.add_argument(
        "--anchor-map",
        action="append",
        default=[],
        help="用于自动推断 Load Bias 的 maps 路径正则，可重复；例如 '^sysmgr-main$'",
    )
    module_group.add_argument(
        "--module-context-id",
        action="append",
        default=[],
        help="该 ELF 允许的 Context ID，可重复，例如 0x2",
    )
    module_group.add_argument(
        "--module-vmid",
        action="append",
        default=[],
        help="该 ELF 允许的 VMID，可重复，例如 0x0",
    )
    module_group.add_argument(
        "--module-el",
        action="append",
        default=[],
        choices=("EL0", "EL1", "EL2", "EL3"),
        help="该 ELF 允许的 Exception Level，可重复",
    )

    parser.add_argument(
        "--module-config",
        help="多模块 JSON 配置；指定后不使用 --elf 等单模块参数",
    )
    parser.add_argument(
        "--page-size",
        default="0x1000",
        help="运行时页大小，默认 0x1000；module config 可覆盖",
    )
    parser.add_argument(
        "--addr2line",
        help="llvm-addr2line/addr2line 路径；默认优先查找 llvm-addr2line",
    )
    parser.add_argument(
        "--address-source",
        dest="address_source",
        action="store_true",
        default=False,
        help="为 address 事件查询 addr2line source；默认关闭以提升速度",
    )
    parser.add_argument(
        "--no-address-source",
        dest="address_source",
        action="store_false",
        help="不为 address 事件查询 addr2line source（默认）",
    )
    parser.add_argument(
        "--only-context-id",
        action="append",
        default=[],
        help="只符号化指定 Context ID，可重复；其他 Address 默认不输出",
    )
    parser.add_argument(
        "--only-el",
        action="append",
        default=[],
        choices=("EL0", "EL1", "EL2", "EL3"),
        help="只符号化指定 EL，可重复",
    )
    parser.add_argument(
        "--progress-bytes",
        type=int,
        default=None,
        help="每处理多少字节输出进度；0 关闭。默认单文件 1000000，批量模式 0",
    )
    parser.add_argument(
        "--atom-mode",
        choices=("commit", "immediate"),
        default="immediate",
        help=(
            "ATOM 处理模式：immediate 在 ATOM 到达时立即恢复（默认，适合 "
            "ptm2human 常见的 ATOM 后合成 Commit-1 输出）；commit 仅用于确认文本中的 "
            "Commit 是真实推测提交事件的情况"
        ),
    )
    parser.add_argument(
        "--max-instructions-per-atom",
        type=int,
        default=65536,
        help="为一个 ATOM 最多扫描多少条指令；超过后停止当前区间，默认 65536",
    )
    parser.add_argument(
        "--instruction-source",
        action="store_true",
        help="对每条恢复指令调用 addr2line；输出更详细但速度显著下降",
    )
    parser.add_argument(
        "--detailed", "--detail",
        dest="detailed",
        action="store_true",
        help=(
            "详细模式：除 instruction_range 外，再输出范围内每条 instruction，"
            "并保留完整 ELF/符号调试字段；trace_stream/trace_info/context "
            "始终作为独立元数据事件输出"
        ),
    )
    parser.add_argument(
        "--include-module",
        action="store_true",
        help=(
            "在事件记录中保留 module 字段；单 ELF 模式默认省略，"
            "多模块配置会自动保留"
        ),
    )
    parser.add_argument(
        "--emit-metadata-events", "--emit_metadata_events",
        dest="emit_metadata_events",
        action="store_true",
        help=(
            "额外输出 discontinuity/commit/pending_atoms_discarded/raw "
            "等调试元数据；默认不写入 JSONL"
        ),
    )
    parser.add_argument(
        "--emit-timestamps", "--emit-timestamp-events",
        dest="emit_timestamp_events",
        action="store_true",
        help="在 JSONL 中额外输出 timestamp 事件；默认不输出以减少噪声",
    )
    batch_group = parser.add_argument_group("批量处理")
    batch_group.add_argument(
        "--workers",
        type=int,
        default=0,
        help="批量模式进程数；0 表示自动，1 表示串行，默认最多使用 8 个进程",
    )
    batch_group.add_argument(
        "--trace-glob",
        action="append",
        default=[],
        help="批量模式文件匹配规则，可重复；默认 '*.txt'，例如 --trace-glob '*.trace'",
    )
    batch_group.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索 --trace-folder 的子目录，并在输出目录保留相对目录结构",
    )
    batch_group.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的批量输出文件；默认跳过",
    )
    batch_group.add_argument(
        "--fail-fast",
        action="store_true",
        help="任一文件处理失败后停止提交新任务；默认继续处理其他文件",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="保留无法识别的原始事件以及被过滤地址",
    )
    return parser


def infer_single_module_label(elf_path: Path) -> str:
    name = elf_path.name
    lowered = name.lower()
    for suffix in (
        ".unstripped.elf",
        ".stripped.elf",
        ".debug.elf",
        ".elf",
        ".debug",
    ):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name or "module"


def modules_from_args(args: argparse.Namespace) -> Tuple[int, List[ModuleRule]]:
    if args.module_config:
        return load_module_config(Path(args.module_config).expanduser())

    if not args.elf:
        raise ValueError("必须指定 --elf，或使用 --module-config")

    elf_path = Path(args.elf).expanduser()
    module_label = infer_single_module_label(elf_path)
    page_size = parse_int(args.page_size, field_name="page_size")
    anchors = list(args.anchor_map)
    if not anchors:
        # 单模块模式下默认使用 ELF 文件名推导 maps 中的主 executable map。
        anchors = [rf"^{re.escape(module_label)}-main$"]

    module = ModuleRule(
        name=module_label,
        elf_path=elf_path,
        load_bias=parse_optional_int(args.load_bias, field_name="load_bias"),
        anchor_maps=anchors,
        context_ids=(
            {parse_int(value, field_name="module_context_id") for value in args.module_context_id}
            or None
        ),
        vmids=(
            {parse_int(value, field_name="module_vmid") for value in args.module_vmid}
            or None
        ),
        els={value.upper() for value in args.module_el} or None,
    )
    return page_size, [module]


def _only_context_ids_from_args(args: argparse.Namespace) -> Optional[Set[int]]:
    return (
        {parse_int(value, field_name="only_context_id") for value in args.only_context_id}
        or None
    )


def _only_els_from_args(args: argparse.Namespace) -> Optional[Set[str]]:
    return {value.upper() for value in args.only_el} or None


def _validate_common_args(args: argparse.Namespace) -> None:
    if args.max_instructions_per_atom <= 0:
        raise ValueError("--max-instructions-per-atom 必须大于 0")
    if args.workers < 0:
        raise ValueError("--workers 不能小于 0")
    if args.progress_bytes is not None and args.progress_bytes < 0:
        raise ValueError("--progress-bytes 不能小于 0")

    # 提前验证 ELF / module config，批量任务不要到 worker 才发现公共配置错误。
    page_size, modules = modules_from_args(args)
    if page_size <= 0:
        raise ValueError("--page-size 必须大于 0")
    for module in modules:
        elf_path = module.elf_path.expanduser().resolve()
        if not elf_path.is_file():
            raise FileNotFoundError(f"ELF 文件不存在：{elf_path}")


def _batch_output_path(
    trace_path: Path,
    trace_folder: Path,
    output_folder: Path,
) -> Path:
    relative = trace_path.relative_to(trace_folder)
    output_name = f"{relative.stem}.symbolized.jsonl"
    return output_folder / relative.parent / output_name


def _discover_trace_files(
    trace_folder: Path,
    patterns: Sequence[str],
    recursive: bool,
    output_folder: Path,
) -> List[Path]:
    found: Dict[str, Path] = {}
    for pattern in patterns:
        iterator = trace_folder.rglob(pattern) if recursive else trace_folder.glob(pattern)
        for candidate in iterator:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            try:
                resolved.relative_to(output_folder)
                # 输出目录位于输入目录内时，避免再次把输出当作输入。
                continue
            except ValueError:
                pass
            if ".symbolized." in resolved.name:
                continue
            found[str(resolved)] = resolved
    return sorted(found.values(), key=lambda path: str(path))


def _run_batch_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """进程池 worker。参数必须保持可 pickle。"""
    import contextlib
    import io
    import traceback

    started = time.perf_counter()
    trace_path = Path(job["trace_path"])
    output_path = Path(job["output_path"])
    temp_output_path = output_path.with_name(
        f"{output_path.name}.part.{os.getpid()}"
    )
    args = argparse.Namespace(**job["args"])
    worker_log = io.StringIO()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            temp_output_path.unlink(missing_ok=True)
        except TypeError:  # Python < 3.8 兼容
            if temp_output_path.exists():
                temp_output_path.unlink()
        with contextlib.redirect_stderr(worker_log):
            page_size, modules = modules_from_args(args)
            needs_addr2line = args.address_source or args.instruction_source
            addr2line_path = (
                resolve_addr2line(args.addr2line) if needs_addr2line else None
            )
            result = process_trace(
                trace_path=trace_path,
                maps_path=Path(args.maps).resolve() if args.maps else None,
                output_path=temp_output_path,
                modules=modules,
                page_size=page_size,
                addr2line_path=addr2line_path,
                address_source=args.address_source,
                verbose=args.verbose,
                progress_bytes=args.progress_bytes,
                only_context_ids=_only_context_ids_from_args(args),
                only_els=_only_els_from_args(args),
                atom_mode=args.atom_mode,
                max_instructions_per_atom=args.max_instructions_per_atom,
                instruction_source=args.instruction_source,
                detailed=args.detailed,
                include_module=args.include_module,
                emit_metadata_events=args.emit_metadata_events,
                emit_timestamp_events=args.emit_timestamp_events,
                emit_summary=False,
            )
        os.replace(temp_output_path, output_path)
        result["output"] = str(output_path)
        return {
            "ok": True,
            "trace": str(trace_path),
            "output": str(output_path),
            "elapsed": time.perf_counter() - started,
            "result": result,
            "log": worker_log.getvalue(),
        }
    except Exception as exc:
        # 防止失败任务留下看起来像成功结果的半文件。
        try:
            if temp_output_path.exists():
                temp_output_path.unlink()
        except OSError:
            pass
        return {
            "ok": False,
            "trace": str(trace_path),
            "output": str(output_path),
            "elapsed": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "log": worker_log.getvalue(),
        }


def _run_single_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    maps_path: Optional[Path],
) -> int:
    trace_path = Path(args.trace).expanduser().resolve()
    if not trace_path.is_file():
        parser.error(f"Trace 文件不存在：{trace_path}")

    output_path = Path(args.out).expanduser().resolve() if args.out else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    page_size, modules = modules_from_args(args)
    needs_addr2line = args.address_source or args.instruction_source
    addr2line_path = resolve_addr2line(args.addr2line) if needs_addr2line else None
    if needs_addr2line:
        if addr2line_path:
            print(f"[INFO] 使用符号化工具：{addr2line_path}", file=sys.stderr)
        else:
            print(
                "[WARN] 未找到 llvm-addr2line/addr2line，将仅输出 ELF/节/符号表信息",
                file=sys.stderr,
            )

    progress_bytes = 1_000_000 if args.progress_bytes is None else args.progress_bytes
    process_trace(
        trace_path=trace_path,
        maps_path=maps_path,
        output_path=output_path,
        modules=modules,
        page_size=page_size,
        addr2line_path=addr2line_path,
        address_source=args.address_source,
        verbose=args.verbose,
        progress_bytes=progress_bytes,
        only_context_ids=_only_context_ids_from_args(args),
        only_els=_only_els_from_args(args),
        atom_mode=args.atom_mode,
        max_instructions_per_atom=args.max_instructions_per_atom,
        instruction_source=args.instruction_source,
        detailed=args.detailed,
        include_module=args.include_module,
        emit_metadata_events=args.emit_metadata_events,
        emit_timestamp_events=args.emit_timestamp_events,
        emit_summary=True,
    )
    return 0


def _run_batch_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    maps_path: Optional[Path],
) -> int:
    trace_folder = Path(args.trace_folder).expanduser().resolve()
    if not trace_folder.is_dir():
        parser.error(f"Trace 目录不存在：{trace_folder}")

    if args.out:
        output_folder = Path(args.out).expanduser().resolve()
    else:
        output_folder = trace_folder.parent / f"{trace_folder.name}_symbolized"
    if output_folder.exists() and not output_folder.is_dir():
        parser.error(f"批量模式 --out 必须是目录：{output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)

    patterns = args.trace_glob or ["*.txt"]
    trace_files = _discover_trace_files(
        trace_folder=trace_folder,
        patterns=patterns,
        recursive=args.recursive,
        output_folder=output_folder,
    )
    if not trace_files:
        parser.error(
            f"目录中未找到匹配文件：folder={trace_folder}, patterns={patterns}"
        )

    jobs: List[Dict[str, Any]] = []
    skipped_existing = 0
    serialized_args = dict(vars(args))
    serialized_args["trace"] = None
    serialized_args["trace_folder"] = str(trace_folder)
    serialized_args["maps"] = str(maps_path) if maps_path else None
    serialized_args["progress_bytes"] = 0 if args.progress_bytes is None else args.progress_bytes

    # 将公共路径转为绝对路径，避免 worker 工作目录差异。
    for key in ("elf", "module_config"):
        value = serialized_args.get(key)
        if value:
            serialized_args[key] = str(Path(value).expanduser().resolve())

    needs_addr2line = args.address_source or args.instruction_source
    resolved_addr2line = (
        resolve_addr2line(args.addr2line) if needs_addr2line else None
    )
    serialized_args["addr2line"] = resolved_addr2line
    if needs_addr2line:
        if resolved_addr2line:
            print(f"[INFO] 批量 worker 使用符号化工具：{resolved_addr2line}", file=sys.stderr)
        else:
            print(
                "[WARN] 未找到 llvm-addr2line/addr2line，批量任务将仅输出 ELF/节/符号表信息",
                file=sys.stderr,
            )

    for trace_path in trace_files:
        output_path = _batch_output_path(
            trace_path, trace_folder, output_folder
        )
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        jobs.append(
            {
                "trace_path": str(trace_path),
                "output_path": str(output_path),
                "args": serialized_args,
            }
        )

    if not jobs:
        print(
            f"[BATCH] 没有需要处理的文件：发现={len(trace_files)}，已存在并跳过={skipped_existing}",
            file=sys.stderr,
        )
        return 0

    cpu_count = os.cpu_count() or 1
    workers = args.workers or min(8, cpu_count, len(jobs))
    workers = max(1, min(workers, len(jobs)))

    print(
        f"[BATCH] input={trace_folder} output={output_folder} "
        f"found={len(trace_files)} queued={len(jobs)} skipped_existing={skipped_existing} "
        f"workers={workers} patterns={patterns} recursive={args.recursive}",
        file=sys.stderr,
    )

    started = time.perf_counter()
    succeeded = 0
    failed = 0
    completed = 0
    failed_results: List[Dict[str, Any]] = []
    aggregate_stats: Counter[str] = Counter()

    def report(result: Dict[str, Any]) -> None:
        nonlocal succeeded, failed, completed
        completed += 1
        if result["ok"]:
            succeeded += 1
            aggregate_stats.update(result.get("result", {}).get("stats", {}))
            print(
                f"[BATCH {completed}/{len(jobs)}] OK "
                f"{result['trace']} -> {result['output']} "
                f"({result['elapsed']:.2f}s)",
                file=sys.stderr,
            )
        else:
            failed += 1
            failed_results.append(result)
            print(
                f"[BATCH {completed}/{len(jobs)}] FAILED "
                f"{result['trace']}: {result['error']}",
                file=sys.stderr,
            )

    if workers == 1:
        for job in jobs:
            result = _run_batch_job(job)
            report(result)
            if args.fail_fast and not result["ok"]:
                break
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_run_batch_job, job): job for job in jobs
            }
            stop_requested = False
            for future in concurrent.futures.as_completed(future_map):
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - worker 进程级故障
                    job = future_map[future]
                    result = {
                        "ok": False,
                        "trace": job["trace_path"],
                        "output": job["output_path"],
                        "elapsed": 0.0,
                        "error": f"worker-process-failure: {type(exc).__name__}: {exc}",
                        "traceback": "",
                        "log": "",
                    }
                report(result)
                if args.fail_fast and not result["ok"]:
                    stop_requested = True
                    for pending in future_map:
                        if not pending.done():
                            pending.cancel()
                    break
            if stop_requested:
                print("[BATCH] fail-fast 已触发，取消尚未开始的任务", file=sys.stderr)

    elapsed = time.perf_counter() - started
    print(
        f"[BATCH SUMMARY] success={succeeded} failed={failed} "
        f"skipped_existing={skipped_existing} elapsed={elapsed:.2f}s "
        f"output={output_folder}",
        file=sys.stderr,
    )
    if aggregate_stats:
        important_keys = (
            "lines", "address", "atom", "commit", "instruction",
            "atom_resolved", "atom_skipped", "unresolved", "flow_anchor",
            "return_target_resolved_by_address",
            "indirect_target_resolved_by_address",
            "exception_target_resolved_by_address",
            "exception_target_unresolved",
        )
        aggregate_text = ", ".join(
            f"{key}={aggregate_stats[key]}"
            for key in important_keys
            if aggregate_stats.get(key, 0)
        )
        if aggregate_text:
            print(f"[BATCH TOTALS] {aggregate_text}", file=sys.stderr)

    if failed_results:
        error_report = output_folder / "batch_errors.log"
        with error_report.open("w", encoding="utf-8") as stream:
            for item in failed_results:
                stream.write(f"=== {item['trace']} ===\n")
                stream.write(f"ERROR: {item['error']}\n")
                if item.get("log"):
                    stream.write("--- worker stderr ---\n")
                    stream.write(item["log"])
                    if not item["log"].endswith("\n"):
                        stream.write("\n")
                if item.get("traceback"):
                    stream.write("--- traceback ---\n")
                    stream.write(item["traceback"])
                    if not item["traceback"].endswith("\n"):
                        stream.write("\n")
                stream.write("\n")
        print(f"[BATCH] 错误详情：{error_report}", file=sys.stderr)

    return 1 if failed else 0


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    print(f"[INFO] trbe_symbolize_sysmgr version={SCRIPT_VERSION}", file=sys.stderr)
    print(f"[INFO] script_path={Path(__file__).resolve()}", file=sys.stderr)

    maps_path = Path(args.maps).expanduser().resolve() if args.maps else None
    if maps_path is not None and not maps_path.is_file():
        parser.error(f"maps 文件不存在：{maps_path}")

    try:
        _validate_common_args(args)
        if args.trace:
            return _run_single_mode(parser, args, maps_path)
        return _run_batch_mode(parser, args, maps_path)
    except KeyboardInterrupt:
        print("\n[ERROR] 用户中断", file=sys.stderr)
        return 130
    except Exception as exc:
        import traceback

        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
