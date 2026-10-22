"""Back-compat shim: the data pipeline lives in `rltrade.data.loaders`."""
from rltrade.data.loaders import DEFAULT_UNIVERSE, load_market  # noqa: F401
