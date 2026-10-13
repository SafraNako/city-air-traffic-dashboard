import os

import pytest

from sehir_panosu import analysis, dashboard, dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def fixture_bytes(name):
    with open(os.path.join(DATA, name), "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def ds():
    return dataset.load()


@pytest.fixture(scope="session")
def prep(ds):
    return analysis.prepare(ds)


@pytest.fixture(scope="session")
def results(ds):
    return analysis.analyze(ds)


@pytest.fixture(scope="session")
def dash(ds, results):
    return dashboard.build(results=results, ds=ds)


@pytest.fixture()
def upstream():
    from upstream import Upstream
    with Upstream() as u:
        yield u
