# Copyright 2025 the LlamaFactory team.
# Python 3.10 compatibility: StrEnum was added in Python 3.11

import sys
from enum import unique

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of StrEnum for Python < 3.11"""
        pass
