"""Compatibility entry point for convert_binary; public defaults and explicit paths apply."""
from importlib import import_module as _import_module
_impl = _import_module("convert_binary")
globals().update({key: value for key, value in vars(_impl).items() if not key.startswith("__")})
run_convert_final = _impl.run_convert_binary
if __name__ == "__main__":
    _impl.main()
