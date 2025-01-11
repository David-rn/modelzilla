import os.path as osp
import pytest


@pytest.fixture
def test_assets_dir():
    cur_dir = osp.dirname(__file__)
    return osp.abspath(osp.join(cur_dir, "assets"))
