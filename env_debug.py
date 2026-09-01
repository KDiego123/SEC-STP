import sys
import platform

print("=== PYTHON ===")
print("Version:", sys.version)
print("Executable:", sys.executable)

print("\n=== PLATFORM ===")
print("OS:", platform.system(), platform.release())
print("Arch:", platform.machine())

print("\n=== LIBRARIES ===")
libs = [
    "numpy",
    "cv2",
    "torch",
    "ultralytics",
    "face_recognition",
    "insightface",
    "onnxruntime"
]

for lib in libs:
    try:
        module = __import__(lib)
        version = getattr(module, "__version__", "unknown")
        path = getattr(module, "__file__", "built-in")
        print(f"{lib}:")
        print(f"  version = {version}")
        print(f"  path    = {path}")
    except Exception as e:
        print(f"{lib}: ERROR -> {e}")

print("\n=== PIP FREEZE (core) ===")
import subprocess
subprocess.run(
    [sys.executable, "-m", "pip", "freeze"],
    stdout=sys.stdout
)
