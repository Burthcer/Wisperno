"""
Centralized design tokens for Wisperno's Qt UI (floating pill + main dashboard),
matching the Wispr Flow-style dark glassmorphism spec. Plain hex strings - usable
directly in QSS stylesheets or wrapped in QColor by callers.
"""

from PySide6.QtGui import QIcon

from src.config import get_base_dir


def get_app_icon() -> QIcon:
    """Resolve assets/icons/wisperno.ico next to the app root (frozen-safe)."""
    icon_path = get_base_dir() / "assets" / "icons" / "wisperno.ico"
    return QIcon(str(icon_path)) if icon_path.exists() else QIcon()

# --- Floating pill ------------------------------------------------------
# Jet-black/purple system (Round 34) - see main dashboard palette below for
# the full rationale; the pill mirrors the same accent so the always-visible
# overlay and the dashboard read as one consistent product, not two.
PILL_BG = "#0B0B0F"
PILL_BG_ALPHA = 0.92
PILL_BORDER = "#1F1F2B"
PILL_TEXT = "#A0A0B0"
PILL_TEXT_PRIMARY = "#EDEDED"

ACCENT_RECORDING = "#EF4444"       # clean red - listening
ACCENT_RECORDING_DIM = "#5C2222"
ACCENT_PROCESSING = "#A855F7"      # radiant violet-purple - transforming
ACCENT_PROCESSING_DIM = "#3B2158"
ACCENT_SUCCESS = "#30D158"         # emerald - injected
ACCENT_ERROR = "#FF9F0A"           # amber - init taking abnormally long / failed

# --- Main dashboard ---------------------------------------------------------
# Jet-black & purple design language (Round 34, replacing the prior indigo/
# obsidian-slate palette per explicit mockups in assets/screenshots/new assets/):
# deeper void backgrounds, a radiant violet-purple accent instead of indigo,
# monospace for technical/numeric content (durations, model paths, metrics),
# clean sans for prose - same structural approach as before (QSS tokens
# cascading through one shared stylesheet), just a new palette.
BG_APP = "#08080C"
BG_SIDEBAR = "#08080C"
SIDEBAR_BORDER = "#1A1A22"
BG_CARD = "#111116"
BG_CARD_HOVER = "#17171F"
BORDER_SUBTLE = "#1F1F2B"
INPUT_BG = "#161620"
INPUT_BORDER = "#262638"
TEXT_PRIMARY = "#EDEDED"
TEXT_SECONDARY = "#71717A"
TEXT_PLACEHOLDER = "#5A5A63"
ACCENT_PRIMARY = "#A855F7"
ACCENT_PRIMARY_ACTIVE = "#C084FC"  # sidebar active-item text/icon tint
ACCENT_PRIMARY_HOVER = "#9333EA"
ACCENT_PRIMARY_WASH = "rgba(168, 85, 247, 0.14)"  # active-nav-item background tint
ACCENT_SECONDARY = "#9061F9"       # muted purple - jargon/category badges
DANGER = "#EF4444"
SUCCESS = "#30D158"

FONT_FAMILY = "Inter, Segoe UI Variable, Segoe UI"
FONT_FAMILY_MONO = "JetBrains Mono, Consolas, monospace"

MAIN_WINDOW_QSS = f"""
QMainWindow, QWidget {{
    background-color: {BG_APP};
    color: {TEXT_PRIMARY};
    font-family: {FONT_FAMILY};
    font-size: 10pt;
}}
QListWidget#sidebarNav {{
    background-color: {BG_SIDEBAR};
    border: none;
    border-right: 1px solid {SIDEBAR_BORDER};
    padding: 8px;
    outline: none;
}}
QListWidget#sidebarNav::item {{
    color: {TEXT_SECONDARY};
    padding: 10px 13px;
    margin: 2px 0 2px 3px;
    height: 40px;
    border-radius: 8px;
    border-left: 3px solid transparent;
}}
QListWidget#sidebarNav::item:selected {{
    background-color: {ACCENT_PRIMARY_WASH};
    border-left: 3px solid {ACCENT_PRIMARY};
    color: {ACCENT_PRIMARY_ACTIVE};
    font-weight: 600;
}}
QListWidget#sidebarNav::item:hover:!selected {{
    background-color: {BG_CARD_HOVER};
}}
QLabel#brand {{
    font-size: 13pt;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    padding: 6px 12px;
}}
QPushButton#liveTranscribeBtn {{
    background-color: {BG_CARD};
    border: 1px solid {DANGER};
    border-radius: 8px;
    margin: 4px 12px 10px 12px;
    padding: 8px 12px;
    color: {TEXT_PRIMARY};
    font-weight: 600;
    text-align: left;
}}
QPushButton#liveTranscribeBtn:hover {{
    background-color: {BG_CARD_HOVER};
}}
QLabel#versionFooter {{
    color: {TEXT_SECONDARY};
    font-size: 8pt;
    padding: 6px 12px;
}}
QFrame#card {{
    /* Apple-style specular sheen: a subtle lighter band at the very top edge,
    fading into the normal card color - a glass/metal chamfer, not a real
    background pattern. */
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1A1A28, stop:0.08 {BG_CARD}, stop:1 {BG_CARD});
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
}}
QFrame#metricCard {{
    background-color: #181824;
    border: 1px solid #262638;
    border-radius: 10px;
}}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox {{
    background-color: {INPUT_BG};
    border: 1px solid {INPUT_BORDER};
    border-radius: 8px;
    padding: 6px 8px;
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_PRIMARY};
    selection-color: white;
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus {{
    border: 1px solid {ACCENT_PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background-color: {INPUT_BG};
    border: 1px solid {BORDER_SUBTLE};
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_PRIMARY};
    outline: none;
}}
QPushButton {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    padding: 7px 16px;
    color: {TEXT_PRIMARY};
}}
QPushButton:hover {{
    background-color: {BG_CARD_HOVER};
    border-color: {ACCENT_PRIMARY};
}}
QPushButton#primary {{
    background-color: {ACCENT_PRIMARY};
    border: none;
    color: white;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background-color: {ACCENT_PRIMARY_HOVER};
}}
QPushButton#danger {{
    color: {DANGER};
}}
QPushButton#filterPill {{
    background-color: transparent;
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 16px;
    padding: 6px 14px;
    color: {TEXT_SECONDARY};
}}
QPushButton#filterPill:hover {{
    color: {TEXT_PRIMARY};
}}
QPushButton#filterPill[active="true"] {{
    background-color: {ACCENT_PRIMARY_WASH};
    border: 1px solid {ACCENT_PRIMARY};
    color: {ACCENT_PRIMARY_ACTIVE};
    font-weight: 600;
}}
QTableWidget, QListWidget, QTreeWidget {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 10px;
    gridline-color: {BORDER_SUBTLE};
    color: {TEXT_PRIMARY};
}}
QHeaderView::section {{
    background-color: {BG_APP};
    color: {TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {BORDER_SUBTLE};
    padding: 6px;
}}
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {BORDER_SUBTLE};
    background-color: {BG_APP};
}}
QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    padding: 8px 4px;
    margin-right: 20px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT_PRIMARY};
}}
QTabBar::tab:selected {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    font-weight: 600;
    border-bottom: 2px solid {ACCENT_PRIMARY};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0px 2px 0px 0px;
}}
QScrollBar::handle:vertical {{
    background: #2E2E42;
    min-height: 24px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT_PRIMARY};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QCheckBox {{ color: {TEXT_PRIMARY}; }}
QSlider::groove:horizontal {{
    height: 4px;
    background: {INPUT_BG};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    height: 4px;
    background: {ACCENT_PRIMARY};
    border-radius: 2px;
}}
QSlider::add-page:horizontal {{
    height: 4px;
    background: {INPUT_BG};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: {TEXT_PRIMARY};
    border: 2px solid {ACCENT_PRIMARY};
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT_PRIMARY};
    border: 2px solid {ACCENT_PRIMARY_HOVER};
}}
QLabel#metric {{ font-size: 16pt; font-weight: 700; color: #EDEDF4; }}
QLabel#metricLabel {{ color: #7E7E94; font-size: 8pt; }}
QLabel#sectionTitle {{
    font-size: 15pt;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    padding: 4px 0 8px 0;
}}
"""

# Round 12 introduced a light Wispr-Flow-styled exception for the Transforms
# tab (WISPR_*/TRANSFORMS_QSS); reverted in the following round back to the
# native obsidian dark theme above, so that palette was removed.
