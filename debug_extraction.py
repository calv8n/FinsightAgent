#!/usr/bin/env python3
"""
Debug script to test tag extraction and LLM responses.
"""

import sys
import os
import re

# Add root to path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

def _extract(tag, text):
    """Extract content between tags (from app.py) - now with flexible spacing."""
    m = re.search(rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

# Test cases
test_responses = [
    # Case 1: Properly formatted response
    """<THOUGHT>
I need to calculate the R&D spend as a percentage of revenue.
</THOUGHT>
<CODE>
import pandas as pd
data = {"AAPL": [6.5, 6.7, 7.8], "MSFT": [12.3, 12.0, 12.4]}
df = pd.DataFrame(data)
print(df)
</CODE>""",

    # Case 2: Missing CODE block
    """<THOUGHT>
I'll calculate this for you.
</THOUGHT>""",

    # Case 3: Tags with different case
    """<thought>
I need to think about this.
</thought>
<code>
print("test")
</code>""",

    # Case 4: Tags with extra spaces
    """<THOUGHT >
Computing the answer.
</THOUGHT >
<CODE >
x = 5 + 3
print(x)
</CODE >""",

    # Case 5: Real LLM response (well-formatted)
    """<THOUGHT>
I should extract the financial data from the retrieved context and compute the ratio.
</THOUGHT>
<CODE>
# Extract R&D and revenue data
apple_rd = 29.9e9  # From filing
apple_revenue = 380.0e9
microsoft_rd = 27.3e9
microsoft_revenue = 198.3e9

# Calculate percentages
apple_pct = (apple_rd / apple_revenue) * 100
microsoft_pct = (microsoft_rd / microsoft_revenue) * 100

print(f"Apple R&D%: {apple_pct:.1f}%")
print(f"Microsoft R&D%: {microsoft_pct:.1f}%")
</CODE>"""
]

print("=" * 80)
print("EXTRACTION TEST SUITE")
print("=" * 80)

for i, response in enumerate(test_responses, 1):
    print(f"\n--- Test Case {i} ---")
    print(f"Input length: {len(response)} chars")
    
    thought = _extract("THOUGHT", response)
    code = _extract("CODE", response)
    final_answer = _extract("FINAL_ANSWER", response)
    
    print(f"THOUGHT: {'✓ Found' if thought else '✗ Not found'}")
    if thought:
        print(f"  → {thought[:80]}")
    
    print(f"CODE: {'✓ Found' if code else '✗ Not found'}")
    if code:
        print(f"  → {len(code)} chars, {code.count(chr(10)) + 1} lines")
        print(f"  → First line: {code.split(chr(10))[0]}")
    
    print(f"FINAL_ANSWER: {'✓ Found' if final_answer else '✗ Not found'}")

print("\n" + "=" * 80)
print("NOW TESTING WITH ACTUAL API...")
print("=" * 80)

try:
    from api.apis import llm_request
    
    system_prompt = """You are a test agent. Respond with:
<THOUGHT>Brief explanation</THOUGHT>
<CODE>print("Hello from test")</CODE>"""
    
    messages = [{"role": "user", "content": "Say hello"}]
    
    print("\nCalling LLM...")
    response = llm_request(system_prompt, messages)
    
    if response is None:
        print("✗ LLM returned None. Check:")
        print("  - GROQ_API_KEY environment variable")
        print("  - Internet connection")
        print("  - API rate limits")
    else:
        print(f"\n✓ LLM Response ({len(response)} chars):")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
        # Extract
        thought = _extract("THOUGHT", response)
        code = _extract("CODE", response)
        
        print(f"\nExtraction Results:")
        print(f"  THOUGHT: {'✓' if thought else '✗'}")
        print(f"  CODE: {'✓' if code else '✗'}")
        
except ImportError as e:
    print(f"✗ Failed to import LLM: {e}")
except Exception as e:
    print(f"✗ Error during API test: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
