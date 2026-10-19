"""Write results as CSV, a Markdown report and a recall-vs-latency chart."""

from __future__ import annotations

import csv
import dataclasses
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rag_benchmark.config import PARAM_ABBREVIATIONS
from rag_benchmark.runner import BenchmarkResult

# Colour follows the backend and is fixed, so a backend looks the same in every
# chart regardless of which others ran. Marker + line style follow the index type.
_BACKEND_COLOURS = {
    "faiss": "#2a78d6",
    "chroma": "#eb6834",
    "milvus": "#1baf7a",
    "pinecone": "#6250d6",
}
_INDEX_STYLES = {"flat": ("D", "-"), "hnsw": ("o", "-"), "ivf_flat": ("s", "--")}
_SURFACE, _INK, _MUTED, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
_PACKAGES = (
    "numpy",
    "faiss-cpu",
    "chromadb",
    "pymilvus",
    "pinecone",
    "google-genai",
    "sentence-transformers",
)


# -- environment ------------------------------------------------------------------


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _ram_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
            return None
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def system_info() -> dict[str, Any]:
    """CPU, RAM, OS, Python and library versions, recorded with every run."""
    ram = _ram_bytes()
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(ram / 1024**3, 1) if ram else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
    }


# -- CSV / JSON -------------------------------------------------------------------


def write_csv(results: Sequence[BenchmarkResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in dataclasses.fields(BenchmarkResult)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for result in results:
            row = dataclasses.asdict(result)
            writer.writerow({k: "" if v is None else v for k, v in row.items()})
    return path


def write_run_json(meta: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return path


# -- Markdown ---------------------------------------------------------------------


def _fmt(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:,.{digits}f}"


def markdown_table(results: Sequence[BenchmarkResult]) -> str:
    """Pipe table of the successful rows."""
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        return "_No successful runs._\n"
    k = ok[0].k
    rag = any(r.hit_rate_at_k is not None for r in ok)
    header = ["Backend", "Config", "Build (s)", f"Recall@{k}", "p50 (ms)", "p95 (ms)"]
    header += ["p99 (ms)", "QPS (1 client)", "QPS (batch)"]
    if rag:
        header = ["Chunker", *header, f"Hit@{k}", f"MRR@{k}"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|"
        + "|".join("---" if i < (3 if rag else 2) else "---:" for i in range(len(header)))
        + "|",
    ]
    for r in ok:
        cells = [
            r.backend,
            r.config,
            _fmt(r.build_s, 2),
            _fmt(r.recall_at_k, 4),
            _fmt(r.p50_ms, 3),
            _fmt(r.p95_ms, 3),
            _fmt(r.p99_ms, 3),
            _fmt(r.qps, 0),
            _fmt(r.batch_qps, 0),
        ]
        if rag:
            cells = [r.chunker, *cells, _fmt(r.hit_rate_at_k, 3), _fmt(r.mrr_at_k, 3)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def not_run_summary(results: Sequence[BenchmarkResult]) -> str:
    """One bullet per backend that was skipped or failed, with the reason."""
    seen: dict[tuple[str, str, str], None] = {}
    for r in results:
        if r.status != "ok":
            seen[(r.backend, r.status, r.note)] = None
    return "".join(f"- **{b}** {status}: {note}\n" for b, status, note in seen)


def write_markdown(
    results: Sequence[BenchmarkResult],
    path: str | Path,
    meta: Mapping[str, Any],
    chart_files: Sequence[str] = (),
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    system = meta.get("system", {})
    first = results[0] if results else None
    lines = ["# rag-benchmark results", ""]
    if first is not None:
        lines += [
            f"- Mode: `{first.mode}`",
            f"- Dataset: {'; '.join(sorted({r.dataset for r in results}))}",
            f"- Queries: {first.n_queries}, k = {first.k}, metric: {meta.get('metric', '')}",
        ]
    if system:
        lines += [
            f"- CPU: {system.get('cpu')} ({system.get('logical_cpus')} logical cores)",
            f"- RAM: {system.get('ram_gb')} GB",
            f"- OS: {system.get('os')}, Python {system.get('python')}",
        ]
    if meta.get("started"):
        lines.append(f"- Started: {meta['started']}")
    lines += ["", markdown_table(results)]
    skipped = not_run_summary(results)
    if skipped:
        lines += ["Not run:", "", skipped]
    for chart in chart_files:
        lines += [f"![recall vs p95 latency]({chart})", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# -- chart ------------------------------------------------------------------------


def _families(rows: Sequence[BenchmarkResult]) -> dict[tuple[str, str], list[int]]:
    """Group rows by (backend, index type); each group is drawn as one curve."""
    groups: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        params = json.loads(row.params or "{}")
        index = str(params.get("index", "hnsw" if row.backend == "chroma" else ""))
        groups.setdefault((row.backend, index), []).append(i)
    return groups


def _varying_params(rows: Sequence[BenchmarkResult], members: Sequence[int]) -> list[str]:
    parsed = [json.loads(rows[m].params or "{}") for m in members]
    keys = dict.fromkeys(key for p in parsed for key in p if key != "index")
    return [k for k in keys if len({json.dumps(p.get(k)) for p in parsed}) > 1]


def _sweep_order(rows: Sequence[BenchmarkResult], members: Sequence[int]) -> list[int]:
    """Order a curve's points by the swept parameter (e.g. ef_search), not by latency,
    so measurement noise cannot make the line zig-zag."""
    varying = _varying_params(rows, members)

    def key(m: int) -> tuple:
        params = json.loads(rows[m].params or "{}")
        values = [params.get(k) for k in varying]
        if all(isinstance(v, int | float) for v in values):
            return (0, *values)
        return (1, rows[m].p95_ms or 0.0)

    return sorted(members, key=key)


def _point_labels(rows: Sequence[BenchmarkResult], members: Sequence[int]) -> dict[int, str]:
    """Label each point with only the parameters that vary within its curve."""
    varying = _varying_params(rows, members)
    if not varying:
        return {}
    labels = {}
    for m in members:
        params = json.loads(rows[m].params or "{}")
        labels[m] = " ".join(f"{PARAM_ABBREVIATIONS.get(k, k)}={params.get(k)}" for k in varying)
    return labels


def plot_recall_vs_latency(
    results: Sequence[BenchmarkResult], path: str | Path, title: str | None = None
) -> Path | None:
    """Recall@k against p95 latency: one curve per backend + index type."""
    from matplotlib.figure import Figure

    ok = [r for r in results if r.status == "ok" and r.p95_ms is not None]
    if not ok:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(10, 6), dpi=150, facecolor=_SURFACE)
    ax = fig.add_subplot(1, 1, 1, facecolor=_SURFACE)
    labels: dict[int, str] = {}
    families = _families(ok)
    for (backend, index), members in families.items():
        members = _sweep_order(ok, members)
        marker, linestyle = _INDEX_STYLES.get(index, ("^", ":"))
        if len(members) == 1:
            linestyle = "none"
        ax.plot(
            [ok[m].p95_ms for m in members],
            [ok[m].recall_at_k for m in members],
            color=_BACKEND_COLOURS.get(backend, _MUTED),
            linestyle=linestyle,
            linewidth=1.4,
            marker=marker,
            markersize=6.5,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.2,
            label=f"{backend} {index}".strip(),
            zorder=3,
        )
        labels.update(_point_labels(ok, members))

    latencies = [r.p95_ms for r in ok if r.p95_ms]
    if latencies and max(latencies) / max(min(latencies), 1e-9) > 5:
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax.margins(x=0.06, y=0.06)

    k = ok[0].k
    if title is None:
        title = f"Recall@{k} vs p95 latency"
        if len(families) == 1:  # a single series is named in the title, not a legend
            title += f" ({' '.join(next(iter(families))).strip()})"
    ax.set_xlabel("p95 query latency (ms, single client)", color=_INK)
    ax.set_ylabel(f"recall@{k}", color=_INK)
    ax.set_title(title, color=_INK, loc="left", fontsize=11)
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8)
    if len(families) > 1:
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    _place_labels(fig, ax, ok, labels)
    fig.savefig(path)
    return path


def _place_labels(
    fig: Any, ax: Any, rows: Sequence[BenchmarkResult], labels: Mapping[int, str]
) -> None:
    """Direct labels beside points, nudged to the first spot that collides with
    nothing already drawn (other labels, markers, the legend) and stays in the axes.
    Points in a crowded spot are left unlabelled instead of overlapping."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.transforms import Bbox

    renderer = FigureCanvasAgg(fig).get_renderer()
    axes_box = ax.get_window_extent(renderer)
    obstacles: list[Any] = []
    pad = 12  # pixels kept clear around every marker, so a label sits by its own point only
    for row in rows:
        x, y = ax.transData.transform((row.p95_ms, row.recall_at_k))
        obstacles.append(Bbox.from_extents(x - pad, y - pad, x + pad, y + pad))
    legend = ax.get_legend()
    if legend is not None:
        obstacles.append(legend.get_window_extent(renderer))

    # Spots right beside the point only: a label never drifts towards another point.
    spots = [(6, 4, "left"), (6, -9, "left"), (-6, 4, "right"), (-6, -9, "right")]
    spots += [(0, 11, "center"), (0, -13, "center")]
    order = sorted(labels, key=lambda i: (-(rows[i].recall_at_k or 0.0), rows[i].p95_ms or 0.0))
    for i in order:
        for dx, dy, align in spots:
            text = ax.annotate(
                labels[i],
                (rows[i].p95_ms, rows[i].recall_at_k),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=align,
                va="center",
                fontsize=7,
                color=_MUTED,
            )
            box = text.get_window_extent(renderer).expanded(1.08, 1.25)  # breathing room
            inside = axes_box.contains(box.x0, box.y0) and axes_box.contains(box.x1, box.y1)
            if inside and not any(box.overlaps(o) for j, o in enumerate(obstacles) if j != i):
                obstacles.append(box)
                break
            # Try the next spot. If none is free the point stays unlabelled rather than
            # overlapping: the legend names the curve and the table has every value.
            text.remove()
