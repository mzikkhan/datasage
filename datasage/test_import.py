# test_import.py
import sys
print("Python path:", sys.path)

try:
    import datasage
    print("✓ datasage imported successfully")
    print("Available:", dir(datasage))
except Exception as e:
    print("✗ Error importing datasage:")
    print(type(e).__name__, ":", e)
    import traceback
    traceback.print_exc()

try:
    from datasage import IndexingEngine
    print("✓ IndexingEngine imported successfully")
except Exception as e:
    print("✗ Error importing IndexingEngine:")
    print(type(e).__name__, ":", e)
    import traceback
    traceback.print_exc()