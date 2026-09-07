"""
Main Application Dashboard for Wisperno (PySide6). Six tabs, each in its own
module: History (history_tab.py), Transforms (transforms_tab.py), Dictionary
(dictionary_tab.py), Snippets (snippets_tab.py), Settings (settings_tab.py),
System Usage (system_usage_tab.py).
"""

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSizePolicy, QScrollArea, QListWidget, QListWidgetItem, QStackedWidget, QPushButton,
)

from src.ui import theme
from src.ui.settings_tab import SettingsTab
from src.ui.history_tab import HistoryTab
from src.ui.transforms_tab import TransformsTab
from src.ui.dictionary_tab import DictionaryTab
from src.ui.snippets_tab import SnippetsTab
from src.ui.system_usage_tab import SystemUsageTab
from src.ui.icons import NAV_ICONS
from src.__version__ import __version__


class MainWindow(QMainWindow):
    def __init__(self, engine, daemon_controller):
        super().__init__()
        self.engine = engine
        self.controller = daemon_controller  # for autostart / pill-visibility toggles

        self.setWindowTitle("Wisperno")
        self.setWindowIcon(theme.get_app_icon())
        self.setStyleSheet(theme.MAIN_WINDOW_QSS)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Explicit, generous bounds rather than trusting the layout's own computed
        # minimum: a single unwrapped long QLabel anywhere in any tab can otherwise
        # push the whole window's effective minimum size out by thousands of pixels
        # (found and fixed several instances of exactly this - see handoff.md).
        # QScrollArea below is the actual fix (decouples content's natural size from
        # the window's), these are just sane, explicit native-window bounds on top.
        self.setMinimumSize(880, 560)
        self.setMaximumSize(16777215, 16777215)  # QWIDGETSIZE_MAX - no artificial ceiling
        self.resize(1020, 680)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 12, 0, 12)
        sidebar_layout.setSpacing(0)
        sidebar_layout.addWidget(QLabel("✦ Wisperno", objectName="brand"))

        # A standalone action button, not a QListWidgetItem: nav items switch
        # the QStackedWidget's page, but Live Transcribe launches a separate
        # floating overlay window (src/ui/live_window.py) instead of a tab.
        live_btn = QPushButton("🔴  Live Transcribe")
        live_btn.setObjectName("liveTranscribeBtn")
        live_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        live_btn.clicked.connect(daemon_controller.toggle_live_transcription)
        sidebar_layout.addWidget(live_btn)

        self.nav = QListWidget()
        self.nav.setObjectName("sidebarNav")
        self.nav.setFrameShape(QFrame.Shape.NoFrame)
        self.nav.setIconSize(QSize(18, 18))
        sidebar_layout.addWidget(self.nav, 1)
        sidebar_layout.addWidget(QLabel(f"Wisperno v{__version__}", objectName="versionFooter"))
        root.addWidget(sidebar, 0)  # stretch 0: never grows/shrinks past its fixed width

        self.stack = QStackedWidget()

        # A QScrollArea around the stack is what actually decouples the window's
        # resizability from any one tab's natural content width - without it, Qt's
        # layout system propagates the widest tab's minimumSizeHint straight up to
        # the window itself, which is what made the window feel "locked" wide.
        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # This scroll area exists purely to decouple the window's resizability from
        # tab content width (see comment above) - it must never actually scroll
        # itself, or a tab with its own internal scroll area (History's card list)
        # ends up nested inside a second, redundant outer scrollbar.
        content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # A hidden scrollbar is still a FUNCTIONAL one in Qt - it silently still
        # scrolls on wheel input even with ScrollBarAlwaysOff above, just without
        # showing a handle. That's the actual root cause of the "scrolling in one
        # tab bleeds into another" bug: with every tab sharing this one outer
        # QScrollArea, scrolling it moves the QStackedWidget itself, and that
        # position is still there (or now hidden with nothing to explain it) the
        # next time a DIFFERENT tab becomes current. Neutering wheel input here
        # forces every scroll gesture down into whichever tab's OWN scroll area
        # (or native scrolling widget, e.g. Dictionary's QTableWidget) is under
        # the cursor - each tab now owns its scroll position, this wrapper owns none.
        content_scroll.wheelEvent = lambda event: event.ignore()
        content_scroll.setWidget(self.stack)
        root.addWidget(content_scroll, 1)  # stretch 1: absorbs all extra width

        self.history_tab = HistoryTab(engine)
        self.transforms_tab = TransformsTab(engine)
        self.dictionary_tab = DictionaryTab(engine)
        self.snippets_tab = SnippetsTab(engine)
        self.settings_tab = SettingsTab(engine, daemon_controller, self)
        self.system_usage_tab = SystemUsageTab(engine)

        for label, page in [
            ("History", self.history_tab),
            ("Transforms", self.transforms_tab),
            ("Dictionary", self.dictionary_tab),
            ("Snippets", self.snippets_tab),
            ("Settings", self.settings_tab),
            ("System Usage", self.system_usage_tab),
        ]:
            item = QListWidgetItem(NAV_ICONS[label](theme.TEXT_SECONDARY), label)
            self.nav.addItem(item)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        self.nav.setCurrentRow(0)

        engine.history_added.connect(lambda _row: self.history_tab.refresh())

    def _on_nav_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        # Each tab owns an independent scroll position now (see content_scroll's
        # wheelEvent override above) - land at the top of whichever tab was just
        # switched to, rather than wherever it happened to be scrolled last time.
        page = self.stack.widget(index)
        if hasattr(page, "reset_scroll"):
            page.reset_scroll()

    def closeEvent(self, event) -> None:
        if self.engine.config.minimize_to_tray:
            event.ignore()
            self.hide()
        elif hasattr(self.controller, "shutdown_application"):
            event.ignore()  # let the controller drive teardown order, not Qt's default close
            self.controller.shutdown_application()
