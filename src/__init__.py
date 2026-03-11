"""
AutoTorch source package.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).parent

PROJECT_ROOT = SRC_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

sys.path.insert(0, str(SRC_DIR))
