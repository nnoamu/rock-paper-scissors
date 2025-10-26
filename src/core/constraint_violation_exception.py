"""
Modul input korlátozás megszegésekor dobódó kivétel.
"""

from typing import List

class ConstraintViolationException(Exception):
    def __init__(self, moduleName: str, violations: List[str]):
        msg="Error: "+moduleName+"'s input constraints aren't met. Violated restrictions:\n"
        for s in violations:
            msg+=" - "+s+"\n"
        super().__init__(msg)