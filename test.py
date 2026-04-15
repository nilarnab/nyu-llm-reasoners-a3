from student.drgrpo_grader import question_only_reward_fn_format_countdown

response_correct = """
Step 1: 74 - 45 = 29
Step 2: 29 + 19 = 48

\\boxed{(74 - 45) + 19    }
"""

gt = {"target": 48, "numbers": [74, 45, 19]}
print("Case 1 (correct):", question_only_reward_fn_format_countdown(response_correct, gt))
