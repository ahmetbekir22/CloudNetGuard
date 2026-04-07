"""pytest konfigürasyonu — CloudNetGuard test suite."""
import sys
import os

# Proje kökünden çalıştırılmayı varsay
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "shared"))
sys.path.insert(0, os.path.join(ROOT, "ai-engine"))
sys.path.insert(0, os.path.join(ROOT, "dns-collector"))
