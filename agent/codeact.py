import json
import time
from .sandbox import execute_code
from api.apis import llm_request
from utils.utils import tag_content

MAX_ITERATIONS = 10
WALL_CLOCK_TIMEOUT_SECONDS = 60


def run_codeact_agent(
    user_query: str,
    system_prompt: str,
    verbose: bool = True,
) -> dict:
    """
    Run complete CodeAct agent loop with exec, observation feedback, and iteration cap.

    Loop Flow:
    1. Agent calls Groq API, responds with <THOUGHT> + <CODE>
    2. We extract code, execute in sandbox, capture output
    3. We feed observation (code output) back to agent
    4. Agent either writes more code or provides <FINAL_ANSWER>
    5. Repeat until final answer or max iterations/timeout

    Args:
        user_query: User's question
        system_prompt: System instruction for agent
        verbose: Print iteration details if True

    Returns:
        dict with keys:
            - query: original user query
            - final_answer: agent's final answer (or None)
            - success: whether agent found an answer
            - iterations: number of loop iterations
            - time_seconds: total time elapsed
            - messages: full conversation history
            - state: final variable state from sandbox
    """

    start_time = time.time()
    iteration_count = 0

    # Initialize message history
    messages = [{"role": "user", "content": user_query}]

    # Initialize persistent state (variables carry over)
    state = {}

    # Final answer holder
    final_answer = None

    if verbose:
        print(f"\n{'='*80}")
        print(f"CodeAct Agent Loop Started")
        print(f"{'='*80}")
        print(f"Query: {user_query}\n")

    # CodeACT LOOP

    while iteration_count < MAX_ITERATIONS:
        iteration_count += 1

        # Check wall-clock timeout
        elapsed = time.time() - start_time
        if elapsed > WALL_CLOCK_TIMEOUT_SECONDS:
            if verbose:
                print(f"\nTIMEOUT: {elapsed:.1f}s > {WALL_CLOCK_TIMEOUT_SECONDS}s")
            break

        if verbose:
            print(f"\n{'-'*80}")
            print(
                f"Iteration {iteration_count}/{MAX_ITERATIONS} | Elapsed: {elapsed:.1f}s"
            )
            print(f"{'-'*80}\n")

        # Agent thinks and writes code

        if verbose:
            print(f"Calling LLM API")

        agent_response = llm_request(system_prompt, messages)

        if agent_response is None:
            if verbose:
                print("API call failed, stopping.")
            break

        if verbose:
            print(f"\nAgent Response:\n{agent_response}\n")

        # Add agent response to message history
        messages.append({"role": "assistant", "content": agent_response})

        # Extract components
        thought = tag_content(agent_response, ["THOUGHT"]).get("THOUGHT", "")
        code = tag_content(agent_response, ["CODE"]).get("CODE", "")
        final_answer_candidate = tag_content(agent_response, ["FINAL_ANSWER"]).get(
            "FINAL_ANSWER", ""
        )

        # Check for final answer

        if final_answer_candidate:
            final_answer = final_answer_candidate
            if verbose:
                print(f"Final Answer Found!\n")
                print(f"Answer: {final_answer}\n")
            break

        # Execute code (if present)

        if not code:
            if verbose:
                print("No <CODE> block found in response. Stopping.\n")
            break

        if verbose:
            print(f"Executing code...\n")
            print(f"Code:\n{code}\n")

        # Execute in sandbox
        output, state, exec_success = execute_code(code, state, iteration_count)

        if verbose:
            print(f"Output:\n{output}")
            if state:
                print(f"State: {list(state.keys())}\n")

        # Feed observation back to agent

        observation = f"""[Code Execution Result]
                        Success: {exec_success}

                        Output:
                        {output}

                        Available Variables: {list(state.keys())}
                        State Summary: {json.dumps({k: str(v)[:100] for k, v in state.items()}, indent=2)}"""

        # Add observation to message history for next iteration
        messages.append({"role": "user", "content": observation})

        if verbose:
            print(f"Observation fed back to agent for next iteration.\n")

    # END OF LOOP

    elapsed_total = time.time() - start_time
    success = final_answer is not None

    if verbose:
        print(f"\n{'='*80}")
        print(f"Loop Completed")
        print(f"{'='*80}")
        print(f"Final Answer: {final_answer or '[None]'}")
        print(f"Success: {success}")
        print(f"Iterations: {iteration_count}")
        print(f"Time: {elapsed_total:.2f}s")
        print(f"Final State: {list(state.keys())}\n")

    return {
        "query": user_query,
        "final_answer": final_answer,
        "success": success,
        "iterations": iteration_count,
        "time_seconds": elapsed_total,
        "messages": messages,
        "state": state,
    }
