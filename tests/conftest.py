import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--module", default=str(ROOT / "tests" / "fixtures" / "KPZ_LO.py"),
                     help="generated KPZ LO module to check against ModelKPZ (default: the vendored fixture)")


@pytest.fixture(scope="session")
def generated_path(request):
    return request.config.getoption("--module")
