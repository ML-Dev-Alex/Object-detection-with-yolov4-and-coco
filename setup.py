import sys
from cx_Freeze import setup, Executable

base = None
if (sys.platform == "win32"):
    base = "Win32GUI"    # Tells the build script to hide the console.

setup(name="Object Detection",
      version="1.0",
      description="Detects objects and displays bounding boxes realtime",
      executables=[Executable("object_detection.py", base=base)]
      )