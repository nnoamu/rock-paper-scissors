"""
Modul input korlátozás megszegésekor dobódó kivétel.
"""

from typing import List

class ConstraintViolationException(Exception):
    def __init__(self, moduleName: str, violations: List[str]):
        msg="Hiba: "+moduleName+" input korlátozásai nem teljesülnek. A sértett korlátozások:\n"
        for s in violations:
            msg+=" - "+s+"\n"
        super().__init__(msg)