"""Import every module in the package.

Most modules have no unit tests, so this is the cheapest check that each
one still loads on every supported Python and dependency set.
"""

import importlib
import pkgutil
import unittest

import aind_dynamic_foraging_behavior_video_analysis as package


class ImportAllModulesTest(unittest.TestCase):
    """Every module in the package imports without error."""

    def test_import_all_modules(self):
        """Walk the package and import each module."""
        names = [
            m.name
            for m in pkgutil.walk_packages(
                package.__path__, package.__name__ + "."
            )
        ]
        self.assertGreater(len(names), 0)
        for name in names:
            with self.subTest(module=name):
                importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
