from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from config import settings

DB_PATH = Path(settings.DB_PATH)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Module-level connection (required by conventions)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# Module-level lock (required by conventions)
db_lock = threading.Lock()
