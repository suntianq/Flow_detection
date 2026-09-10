import unittest

from tools.ghidra.ghidra_common import bool_int, java_iter, safe_call


class _Example:
    def value(self):
        return 7

    def broken(self):
        raise RuntimeError("boom")


class GhidraCommonTest(unittest.TestCase):
    def test_safe_call(self) -> None:
        value = _Example()
        self.assertEqual(safe_call(value, "value"), 7)
        self.assertEqual(safe_call(value, "broken", "fallback"), "fallback")
        self.assertEqual(safe_call(None, "value", "fallback"), "fallback")

    def test_java_iter_accepts_python_iterables(self) -> None:
        self.assertEqual(list(java_iter([1, 2, 3])), [1, 2, 3])

    def test_bool_int(self) -> None:
        self.assertEqual(bool_int(True), 1)
        self.assertEqual(bool_int(False), 0)


if __name__ == "__main__":
    unittest.main()
