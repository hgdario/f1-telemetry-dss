"""
conftest.py — raíz de src/

Asegura que el directorio src/ esté en sys.path para que los tests puedan
importar los módulos de la aplicación con sus imports relativos al paquete
(p.ej. ``from dinamica_vehicular.GGDiagram import _calculate_g_forces``).
"""

import os
import sys

_SRC = os.path.dirname(__file__)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
