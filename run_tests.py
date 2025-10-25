import unittest
from tests import GrepTest

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(GrepTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.failures:
        print("Failures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    if result.errors:
        print("Errors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
