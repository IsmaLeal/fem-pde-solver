import importlib

def test_cli_import():
    module = importlib.import_module("src.cli")
    assert hasattr(module, "main")
