"""Development settings for MSA Port Twin."""

import os

os.environ.setdefault("MODE", "dev")

from .settings import *

DEBUG = True
