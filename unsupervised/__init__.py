"""Compatibility shim for the top-level unsupervised package."""

from repo_bootstrap import ensure_src_on_path

ensure_src_on_path()

from catfood_unsupervised.compat.unsupervised import *  # noqa: F401,F403
from catfood_unsupervised.compat.unsupervised import __all__ as _compat_all

__all__ = list(_compat_all)
