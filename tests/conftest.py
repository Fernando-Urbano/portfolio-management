import pytest
import pandas as pd

@pytest.fixture(scope="session", autouse=True)
def setup_pandas_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

def test_setup_pandas_display():
    assert pd.get_option('display.max_columns') is None
    assert pd.get_option('display.expand_frame_repr') is False
