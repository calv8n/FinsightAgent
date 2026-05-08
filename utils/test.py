from agent.codeact import MAX_ITERATIONS, run_codeact_agent
from utils.prompts import CODEACT_SYSTEM_PROMPT


def test_basic_math():
    """Test 1: Simple math problem (should take 1-2 iterations)."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Math (Sum of 1 to 100)")
    print("=" * 80)

    query = "What is the sum of all integers from 1 to 100?"
    result = run_codeact_agent(query, CODEACT_SYSTEM_PROMPT, verbose=True)

    assert result["success"], "Should find answer"
    assert result["final_answer"] is not None
    assert "5050" in result["final_answer"], "Answer should contain 5050"
    assert result["iterations"] <= 3, "Should solve in 2-3 iterations"

    print("\nTest 1 PASSED\n")
    return result


def test_multi_step_reasoning():
    """Test 2: Multi-step problem requiring state persistence (median)."""
    print("\n" + "=" * 80)
    print("TEST 2: Multi-Step Reasoning (Median)")
    print("=" * 80)

    query = "Find the median of the list [3, 1, 4, 1, 5, 9, 2, 6]. Show your work."
    result = run_codeact_agent(query, CODEACT_SYSTEM_PROMPT, verbose=True)

    assert result["success"], "Should find answer"
    assert result["final_answer"] is not None
    assert "3.5" in result["final_answer"], "Answer should contain 3.5"
    assert result["iterations"] >= 2, "Should use at least 2 iterations to show work"

    print("\nTest 2 PASSED\n")
    return result


def test_state_persistence():
    """Test 3: Verify variables persist across iterations."""
    print("\n" + "=" * 80)
    print("TEST 3: State Persistence Across Iterations")
    print("=" * 80)

    query = """explain me how a sliding window algorithm works and how would i implement it in a rag ingestion pipeline?
"""
    result = run_codeact_agent(query, CODEACT_SYSTEM_PROMPT, verbose=True)
    return result


def test_multi_hop_question():
    """Test 4: Complex multi-hop question (factorial with verification)."""
    print("\n" + "=" * 80)
    print("TEST 4: Multi-Hop Question (Factorial)")
    print("=" * 80)

    query = """
    Calculate 10 factorial using a loop.
    Then verify the result by computing 10 × 9 × 8 × 7 × 6 × 5 × 4 × 3 × 2 × 1 manually.
    Are they equal?
    """
    result = run_codeact_agent(query, CODEACT_SYSTEM_PROMPT, verbose=True)

    assert result["success"], "Should find answer"
    assert result["final_answer"] is not None
    assert "3628800" in result["final_answer"], "10! = 3,628,800"
    assert result["iterations"] >= 2, "Should use multiple iterations for verification"

    print("\nTest 4 PASSED\n")
    return result


def test_iteration_cap():
    """Test 5: Verify iteration cap is enforced."""
    print("\n" + "=" * 80)
    print("TEST 5: Iteration Cap (Max 10)")
    print("=" * 80)

    # This shouldn't actually hit the cap for a simple query,
    # but we verify the cap exists
    query = "What is 2 + 2?"
    result = run_codeact_agent(query, CODEACT_SYSTEM_PROMPT, verbose=True)

    assert (
        result["iterations"] <= MAX_ITERATIONS
    ), f"Should not exceed {MAX_ITERATIONS} iterations"

    print("\nTest 5 PASSED\n")
    return result


def test_error_handling():
    """Test 6: Agent handles code errors gracefully."""
    print("\n" + "=" * 80)
    print("TEST 6: Error Handling (Agent recovers from errors)")
    print("=" * 80)

    query = "Write code that has a bug on the first try, catch it, fix it, then compute 5+5."
    result = run_codeact_agent(query, CODEACT_SYSTEM_PROMPT, verbose=True)

    # Agent should still succeed despite encountering an error
    assert result["success"] or result["iterations"] > 1, "Should attempt recovery"

    print("\nTest 6 PASSED\n")
    return result
