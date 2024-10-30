import pytest

def test_import():
    try:
        from pyscfadlib import lapack as lp
    except ImportError as e:
        raise ImportError(e)
