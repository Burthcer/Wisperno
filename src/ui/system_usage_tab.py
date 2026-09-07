"""
Dedicated System Usage analytics tab: live VRAM/RAM telemetry (via
src.system_monitor's PDH-based per-process GPU reader) plus a card showing
which STT/LLM engines are active and their last measured throughput/latency.
Split out from Settings -> Advanced, which is for configuration, not monitoring.
"""

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QProgressBar, QStyleFactory

from src.ui import theme
from src.ui.scroll_utils import make_scrollable_layout
from src.system_monitor import SystemMonitor

# A flat qlineargradient QSS on QProgressBar::chunk still gets tinted by
# Windows' native "vista" style overlay (its own glossy highlight shading can
# shift a pure purple toward pink) - forcing Fusion is what actually makes
# the stylesheet color the whole rendered bar, not just a base layer under a
# native gloss.
_BAR_STYLE = QStyleFactory.create("Fusion")

_VRAM_GRADIENT = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7C3AED, stop:1 #A855F7)"
_RAM_GRADIENT = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4F46E5, stop:1 #818CF8)"


def _card(title: str) -> tuple:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    heading = QLabel(title)
    heading.setStyleSheet(f"font-size: 12pt; font-weight: 700; color: {theme.TEXT_PRIMARY};")
    layout.addWidget(heading)
    return frame, layout


class SystemUsageTab(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.monitor = SystemMonitor()

        layout = make_scrollable_layout(self)
        layout.setContentsMargins(28, 22, 28, 22)
        title = QLabel("System Usage")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            f'<span style="color:{theme.TEXT_SECONDARY}">Live GPU/RAM telemetry and active-engine status.</span>'
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Transition indicator, shown while a preset switch (or the app's own
        # first startup) is reloading Whisper/LLM - see settings_tab.py's
        # matching banner for why the wording is generic, not preset-specific.
        self.reload_banner = QLabel("⟳ Loading engine (Whisper + LLM) - this can take a few seconds...")
        self.reload_banner.setWordWrap(True)
        self.reload_banner.setStyleSheet(
            f"background-color: {theme.INPUT_BG}; color: {theme.ACCENT_PRIMARY}; "
            f"border-radius: 6px; padding: 6px 10px; font-weight: 600;"
        )
        self.reload_banner.setVisible(False)
        layout.addWidget(self.reload_banner)
        if hasattr(engine, "state_changed"):
            engine.state_changed.connect(self._on_engine_state_changed)

        # --- GPU VRAM card ---
        vram_card, vram_layout = _card("GPU VRAM")
        self.vram_bar = QProgressBar()
        self.vram_bar.setRange(0, 100)
        self.vram_bar.setTextVisible(False)
        self.vram_bar.setFixedHeight(14)
        self.vram_bar.setStyle(_BAR_STYLE)
        self.vram_bar.setStyleSheet(
            f"QProgressBar {{ background-color: {theme.INPUT_BG}; border-radius: 7px; }}"
            f"QProgressBar::chunk {{ background: {_VRAM_GRADIENT}; border-radius: 7px; }}"
        )
        self.vram_label = QLabel("Reading...")
        self.vram_label.setWordWrap(True)
        self.vram_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        self.vram_sub_label = QLabel("")
        self.vram_sub_label.setWordWrap(True)
        self.vram_sub_label.setStyleSheet(
            f"color: {theme.ACCENT_PRIMARY}; font-weight: 600; font-family: {theme.FONT_FAMILY_MONO}; padding-top: 2px;"
        )
        vram_layout.addWidget(self.vram_bar)
        vram_layout.addWidget(self.vram_label)
        vram_layout.addWidget(self.vram_sub_label)
        layout.addWidget(vram_card)

        # --- System RAM card ---
        ram_card, ram_layout = _card("System RAM")
        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setTextVisible(False)
        self.ram_bar.setFixedHeight(14)
        self.ram_bar.setStyle(_BAR_STYLE)
        self.ram_bar.setStyleSheet(
            f"QProgressBar {{ background-color: {theme.INPUT_BG}; border-radius: 7px; }}"
            f"QProgressBar::chunk {{ background: {_RAM_GRADIENT}; border-radius: 7px; }}"
        )
        self.ram_label = QLabel("Reading...")
        self.ram_label.setWordWrap(True)
        self.ram_label.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")
        self.ram_sub_label = QLabel("")
        self.ram_sub_label.setWordWrap(True)
        self.ram_sub_label.setStyleSheet(
            f"color: {theme.ACCENT_PRIMARY}; font-weight: 600; font-family: {theme.FONT_FAMILY_MONO}; padding-top: 2px;"
        )
        ram_layout.addWidget(self.ram_bar)
        ram_layout.addWidget(self.ram_label)
        ram_layout.addWidget(self.ram_sub_label)
        layout.addWidget(ram_card)

        # --- Engine telemetry card ---
        engine_card, engine_layout = _card("Active Engines")
        self.stt_label = QLabel("...")
        self.llm_label = QLabel("...")
        self.perf_label = QLabel("...")
        for lbl in (self.stt_label, self.llm_label, self.perf_label):
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; padding-top: 4px;")
            engine_layout.addWidget(lbl)
        layout.addWidget(engine_card)

        layout.addStretch()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(2000)
        self.refresh()

    def reset_scroll(self) -> None:
        self._scroll_area.verticalScrollBar().setValue(0)

    def _on_engine_state_changed(self, state: str) -> None:
        self.reload_banner.setVisible(state == "loading")

    def refresh(self) -> None:
        snap = self.monitor.snapshot()
        gpu_total_gb = snap["gpu_total_mb"] / 1024
        app_vram_gb = snap["app_vram_mb"] / 1024
        gpu_used_gb = snap["gpu_used_mb"] / 1024
        ram_total_gb = snap["ram_total_mb"] / 1024
        ram_used_gb = snap["ram_used_mb"] / 1024
        app_ram_gb = snap["app_ram_mb"] / 1024
        ram_free_gb = max(ram_total_gb - ram_used_gb, 0.0)

        if gpu_total_gb > 0:
            pct = int(gpu_used_gb / gpu_total_gb * 100)
            self.vram_bar.setValue(pct)
            self.vram_label.setText(
                f"Wisperno: {app_vram_gb:.1f} GB  |  Total VRAM: {gpu_used_gb:.1f} / {gpu_total_gb:.1f} GB ({pct}%)"
            )
            util_txt = f"{snap['gpu_util_pct']}%" if snap["gpu_util_pct"] is not None else "n/a"
            temp_txt = f"{snap['gpu_temp_c']}°C" if snap["gpu_temp_c"] is not None else "n/a"
            self.vram_sub_label.setText(f"GPU Utilization: {util_txt}  |  Temperature: {temp_txt}")
        else:
            self.vram_bar.setValue(0)
            self.vram_label.setText("No NVIDIA GPU detected.")
            self.vram_sub_label.setText("")

        pct = int(ram_used_gb / ram_total_gb * 100) if ram_total_gb > 0 else 0
        self.ram_bar.setValue(pct)
        self.ram_label.setText(
            f"System Used: {ram_used_gb:.1f} / {ram_total_gb:.1f} GB ({pct}%)  |  Free: {ram_free_gb:.1f} GB"
        )
        self.ram_sub_label.setText(f"Wisperno Process: {app_ram_gb:.1f} GB RAM")

        self._refresh_engine_card()

    def _refresh_engine_card(self) -> None:
        transcriber = self.engine.transcriber
        transformer = self.engine.transformer

        if transcriber is not None:
            cfg = transcriber.config
            self.stt_label.setText(f"STT: Whisper {cfg.model_name} [{cfg.compute_type.upper()}]")
        else:
            self.stt_label.setText("STT: not loaded yet")

        if transformer is not None and transformer.llm is not None:
            total = transformer.active_model_layers
            offload = transformer.config.n_gpu_layers
            if offload < 0:
                pct_txt = "100% GPU"  # llama.cpp's own "all layers" convention
            elif total:
                pct_txt = "100% GPU" if offload >= total else f"{offload}/{total} layers GPU"
            else:
                pct_txt = f"{offload} layers GPU"
            self.llm_label.setText(f"LLM: {transformer.active_model_name} ({pct_txt})")
        else:
            self.llm_label.setText("LLM: not loaded yet")

        rows = self.engine.db.list_history(limit=10)
        if rows:
            last_latency_s = rows[0]["latency_ms"] / 1000.0
            speeds = []
            for r in rows:
                words = len(r["polished_transcript"].split())
                secs = r["latency_ms"] / 1000.0
                if secs > 0 and words > 0:
                    speeds.append(words * 1.3 / secs)  # ~1.3 tokens/word, approximate
            avg_tps = sum(speeds) / len(speeds) if speeds else 0.0
            self.perf_label.setText(
                f"Last pipeline latency: {last_latency_s:.2f}s  |  Avg throughput (last {len(rows)}): ~{avg_tps:.1f} tok/s (approx.)"
            )
        else:
            self.perf_label.setText("No dictation recorded yet this session.")
