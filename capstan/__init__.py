"""Capstan compatibility package.

Capstan is the public product/CLI name. The implementation still lives in
``modelctl`` during the compatibility window.
"""

from modelctl import LEGACY_CLI_NAME, PRODUCT_NAME, __version__

__all__ = ["LEGACY_CLI_NAME", "PRODUCT_NAME", "__version__"]
