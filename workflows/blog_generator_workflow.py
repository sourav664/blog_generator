import os
import sys
import re
from datetime import datetime
from typing import Optional
from langgraph.types import Send
from jinja2 import Template
from pathlib import Path


curr_dir = Path(__file__).resolve().parents[0]
print(curr_dir)