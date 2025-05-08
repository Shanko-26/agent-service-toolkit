try:
    from src.data.parser import parse_mdf_file
    print("Import successful!")
    print(f"Function exists: {callable(parse_mdf_file)}")
except ImportError as e:
    print(f"Import failed: {e}") 