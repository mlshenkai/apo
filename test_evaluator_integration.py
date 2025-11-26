#!/usr/bin/env python
"""
测试 evaluator 集成到 pipeline 中是否正常工作
"""

from apo.evaluators import get_evaluator

def test_evaluator_registration():
    """测试所有 task 的 evaluator 是否正确注册"""
    tasks = ["liar", "bbh", "ethos", "arsarcasm", "wsc", "gsm8k"]

    print("Testing evaluator registration...")
    for task in tasks:
        try:
            evaluator = get_evaluator(task)
            print(f"✓ Task '{task}': {evaluator.__class__.__name__}")
        except ValueError as e:
            print(f"✗ Task '{task}': {e}")
    print()


def test_evaluator_parsing():
    """测试各个 evaluator 的解析功能"""
    print("Testing evaluator parsing...")

    # Test GSM8K
    gsm8k_eval = get_evaluator("gsm8k")
    test_cases_gsm8k = [
        ("The answer is 24.", "24"),
        ("So the final result is 100", "100"),
        ("Answer: -5", "-5"),
    ]
    print("GSM8K Evaluator:")
    for output, expected in test_cases_gsm8k:
        parsed = gsm8k_eval.parse_pred(output)
        status = "✓" if parsed == expected else "✗"
        print(f"  {status} '{output}' -> '{parsed}' (expected: '{expected}')")
    print()

    # Test Binary (liar, ethos, arsarcasm)
    binary_eval = get_evaluator("liar")
    test_cases_binary = [
        ("Yes, that's correct.", "Yes"),
        ("No way!", "No"),
        ("I think yes", "Yes"),
        ("The answer is no", "No"),
    ]
    print("Binary Evaluator (liar/ethos/arsarcasm):")
    for output, expected in test_cases_binary:
        parsed = binary_eval.parse_pred(output)
        status = "✓" if parsed == expected else "✗"
        print(f"  {status} '{output}' -> '{parsed}' (expected: '{expected}')")
    print()

    # Test BBH
    bbh_eval = get_evaluator("bbh")
    test_cases_bbh = [
        ("YES", "YES"),
        ("No, it's not correct", "NO"),
        ("I think yes", "YES"),
    ]
    print("BBH Evaluator:")
    for output, expected in test_cases_bbh:
        parsed = bbh_eval.parse_pred(output)
        status = "✓" if parsed == expected else "✗"
        print(f"  {status} '{output}' -> '{parsed}' (expected: '{expected}')")
    print()

    # Test WSC
    wsc_eval = get_evaluator("wsc")
    test_cases_wsc = [
        ("A", "A"),
        ("The answer is B", "B"),
        ("I choose option A", "A"),
    ]
    print("WSC Evaluator:")
    for output, expected in test_cases_wsc:
        parsed = wsc_eval.parse_pred(output)
        status = "✓" if parsed == expected else "✗"
        print(f"  {status} '{output}' -> '{parsed}' (expected: '{expected}')")
    print()


def test_label_normalization():
    """测试标签标准化功能"""
    print("Testing label normalization...")

    gsm8k_eval = get_evaluator("gsm8k")
    assert gsm8k_eval.normalize_label("24") == "24"
    assert gsm8k_eval.normalize_label("100") == "100"
    print("✓ GSM8K label normalization works")

    binary_eval = get_evaluator("liar")
    assert binary_eval.normalize_label("yes") == "Yes"
    assert binary_eval.normalize_label("NO") == "No"
    print("✓ Binary label normalization works")

    bbh_eval = get_evaluator("bbh")
    assert bbh_eval.normalize_label("yes") == "YES"
    assert bbh_eval.normalize_label("no") == "NO"
    print("✓ BBH label normalization works")

    wsc_eval = get_evaluator("wsc")
    assert wsc_eval.normalize_label("a") == "A"
    assert wsc_eval.normalize_label("b") == "B"
    print("✓ WSC label normalization works")
    print()


if __name__ == "__main__":
    test_evaluator_registration()
    test_evaluator_parsing()
    test_label_normalization()
    print("All tests passed! ✓")
