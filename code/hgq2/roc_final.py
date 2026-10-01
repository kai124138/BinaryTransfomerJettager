"""Compatibility entry point for evaluate_roc; public defaults and explicit paths apply."""
from importlib import import_module as _import_module
_impl = _import_module("evaluate_roc")
globals().update({key: value for key, value in vars(_impl).items() if not key.startswith("__")})
if __name__ == "__main__":
    _impl.main()
