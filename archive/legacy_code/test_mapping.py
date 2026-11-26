"""
Test Step3 mapping functionality.
Verifies that CATEGORY_MAPPING correctly maps CSV (source_step, category, subcategory) to Penalty.json (step, category_key).
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directly define CATEGORY_MAPPING
# File path contains spaces, so we define it here
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Read Penalty.json
penalty_json_path = os.path.join(base_dir, "Penalty", "Penalty.json")
with open(penalty_json_path, "r", encoding="utf-8") as f:
    penalty_data = json.load(f)

# Define CATEGORY_MAPPING (same as in test_penalty_evaluator.py)
CATEGORY_MAPPING = {
    # Step1 mappings
    ("Step1", "hard_prompt", "math"): ("Step1", "math"),
    ("Step1", "hard_prompt", "coding"): ("Step1", "coding"),
    ("Step1", "creative_writing", "creative_writing"): ("Step1", "creative_writing"),
    ("Step1", "LiveCodeBench", "easy"): ("Step1", "LiveCodeBench"),
    ("Step1", "LiveCodeBench", "medium"): ("Step1", "LiveCodeBench"),
    ("Step1", "LiveCodeBench", "hard"): ("Step1", "LiveCodeBench"),
    ("Step1", "GPQA", None): ("Step1", "GPQA"),
    
    # Step2 mappings
    ("Step2", "short_prompt", "simple"): ("Step2", "short_prompt_simple"),
    ("Step2", "long_prompt", "simple"): ("Step2", "long_prompt_simple"),
    ("Step2", "short_prompt", "complex"): ("Step2", "short_prompt_complex"),
    ("Step2", "long_prompt", "complex"): ("Step2", "long_prompt_complex"),
    
    # Step3 mappings
    # Supports two subcategory formats:
    # 1. Old format: (category, subcategory) -> (json_step, json_category_key)
    # 2. New format: subcategory directly matches Penalty.json key names (e.g., workflow_cs_simple, workflow_cs_conflict, workflow_cs_prediction)
    ("Step3", "customer_support", "information_query"): ("Step3", "workflow_cs_simple"),
    ("Step3", "customer_support", "workflow_cs_simple"): ("Step3", "workflow_cs_simple"),
    ("Step3", "customer_support", "workflow_cs_conflict"): ("Step3", "workflow_cs_conflict"),
    ("Step3", "customer_support", "workflow_cs_prediction"): ("Step3", "workflow_cs_prediction"),
    ("Step3", "research", "investigation"): ("Step3", "workflow_research"),
    ("Step3", "translation", "multilingual"): ("Step3", "workflow_translation"),
}


def get_prompt_mapping(source_step, category, subcategory):
    """Get the prompt mapping from CSV fields to Penalty.json keys."""
    cat = category.strip() if category else ""
    sub = subcategory.strip() if subcategory else ""
    
    # Special handling for GPQA
    if source_step == "Step1" and cat.lower() == "gpqa":
        return ("Step1", "GPQA")
    
    # Try exact match
    key = (source_step, cat, sub if sub else None)
    if key in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[key]
    
    # Try case-insensitive match
    for map_key, map_val in CATEGORY_MAPPING.items():
        if map_key[0] != source_step:
            continue
        if map_key[1].lower() != cat.lower():
            continue
        if map_key[2] is None and (sub == "" or sub is None):
            return map_val
        if map_key[2] and map_key[2].lower() == sub.lower():
            return map_val
    
    return None


def test_mapping():
    """Test the mapping function."""
    print("=" * 80)
    print("  Step3 Mapping Test")
    print("=" * 80)
    
    # Define test cases
    test_cases = [
        # (source_step, category, subcategory, expected_mapping, description)
        ("Step3", "customer_support", "workflow_cs_simple", ("Step3", "workflow_cs_simple"), "CS Simple Info Query (new format)"),
        ("Step3", "customer_support", "workflow_cs_conflict", ("Step3", "workflow_cs_conflict"), "CS Conflict Detection (new format)"),
        ("Step3", "customer_support", "workflow_cs_prediction", ("Step3", "workflow_cs_prediction"), "CS Time Prediction (new format)"),
        ("Step3", "customer_support", "information_query", ("Step3", "workflow_cs_simple"), "CS Simple Info Query (old format)"),
        ("Step3", "research", "investigation", ("Step3", "workflow_research"), "Research Investigation"),
        ("Step3", "translation", "multilingual", ("Step3", "workflow_translation"), "Translation"),
    ]
    
    all_passed = True
    
    for source_step, category, subcategory, expected, description in test_cases:
        result = get_prompt_mapping(source_step, category, subcategory)
        
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        
        print(f"\n[{status}] {description}")
        print(f"  Input: ({source_step}, {category}, {subcategory})")
        print(f"  Expected: {expected}")
        print(f"  Actual: {result}")
        
        # Verify prompt exists in Penalty.json
        if result:
            json_step, json_cat_key = result
            if json_step in penalty_data and json_cat_key in penalty_data[json_step]:
                prompt_data = penalty_data[json_step][json_cat_key]
                print(f"  Penalty.json: OK - Prompt found")
                print(f"    - system_prompt length: {len(prompt_data.get('system_prompt', ''))} chars")
                print(f"    - user_prompt_template length: {len(prompt_data.get('user_prompt_template', ''))} chars")
            else:
                print(f"  Penalty.json: FAIL - Prompt NOT found (step={json_step}, key={json_cat_key})")
                all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("  All tests PASSED!")
    else:
        print("  SOME TESTS FAILED!")
    print("=" * 80)
    
    # Show all available Step3 categories in Penalty.json
    print("\n  All available Step3 categories in Penalty.json:")
    step3_data = penalty_data.get("Step3", {})
    for key in step3_data.keys():
        print(f"    - {key}")
    
    return all_passed


if __name__ == "__main__":
    success = test_mapping()
    sys.exit(0 if success else 1)
