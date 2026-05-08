from utils.test import *


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CODEACT AGENT - COMPLETE EXEC LOOP TEST SUITE")
    print("=" * 80)

    tests = [
        # ("Basic Math", test_basic_math),
        # ("Multi-Step Reasoning", test_multi_step_reasoning),
        ("State Persistence", test_state_persistence),
        # ("Multi-Hop Question", test_multi_hop_question),
        # ("Iteration Cap", test_iteration_cap),
        # ("Error Handling", test_error_handling),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, True, None))
        except AssertionError as e:
            print(f"\nTest FAILED: {e}\n")
            results.append((name, False, str(e)))
        except Exception as e:
            print(f"\nTest ERROR: {e}\n")
            import traceback

            traceback.print_exc()
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed, error in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")

    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} passed\n")

    return passed_count == total_count


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
