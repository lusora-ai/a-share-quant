"""
Symbol Conversion Utilities between A-share Market Formats and Qlib Canonical Format.
Examples:
  A-share: "600000.SH", "000001.SZ", "000300.SH"
  Qlib:    "SH600000", "SZ000001", "SH000300"
"""
import re
from typing import Union, List

def to_qlib_symbol(symbol: str) -> str:
    """
    Convert A-share standard code to Qlib canonical symbol.
    600000.SH -> SH600000
    000001.SZ -> SZ000001
    sh600000  -> SH600000
    SH000300  -> SH000300
    """
    s = symbol.strip().upper()
    if "." in s:
        code, market = s.split(".", 1)
        return f"{market}{code}"
    # If already in format SH600000 / SZ000001
    if s.startswith(("SH", "SZ", "BJ")):
        return s
    # Fallback heuristic based on prefix if only 6 digits
    if re.match(r"^\d{6}$", s):
        if s.startswith(("6", "9", "5")):
            return f"SH{s}"
        elif s.startswith(("0", "3", "1", "2")):
            return f"SZ{s}"
        elif s.startswith(("4", "8")):
            return f"BJ{s}"
    return s

def from_qlib_symbol(qlib_symbol: str) -> str:
    """
    Convert Qlib canonical symbol to standard A-share ts_code.
    SH600000 -> 600000.SH
    SZ000001 -> 000001.SZ
    """
    s = qlib_symbol.strip().upper()
    if s.startswith(("SH", "SZ", "BJ")):
        market = s[:2]
        code = s[2:]
        return f"{code}.{market}"
    if "." in s:
        return s
    return s
