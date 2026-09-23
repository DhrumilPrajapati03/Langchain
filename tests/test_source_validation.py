"""Repository-level tests that run without third-party service credentials."""

from __future__ import annotations

import py_compile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ("Projects", "week-1", "week-4")


class SourceValidationTests(unittest.TestCase):
    def test_python_modules_compile(self) -> None:
        """Every tracked application module must be valid Python syntax."""
        source_files = [
            source_file
            for directory in SOURCE_DIRECTORIES
            for source_file in (REPOSITORY_ROOT / directory).rglob("*.py")
            if "__pycache__" not in source_file.parts
        ]

        self.assertGreater(len(source_files), 0, "No Python source files were found")

        for source_file in source_files:
            with self.subTest(source_file=source_file.relative_to(REPOSITORY_ROOT)):
                py_compile.compile(source_file, doraise=True)
