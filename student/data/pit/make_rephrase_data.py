import argparse
import json

from student.sec_7.dataloader_normal import format_prompt


def convert_jsonl_to_json(input_path: str, output_path: str):
    entries = []

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            original_question = record["original_question"]
            original_raw = record["original_raw"]
            original_answer = record["original_answer"]
            modified = record.get("modified_questions", {})
            adversarials = modified.get("adverserials", [])

            # Original entry
            entries.append({
                "question": original_question,
                "raw": original_raw,
                "answer": original_answer,
                "original_question": original_question,
                "is_adverserial": False,
            })

            # Adversarial entries
            for adv_question in adversarials:
                rephrased_raw = (
                    f"Let me rephrase. "
                    f"<rephrase>{format_prompt(original_raw)}</rephrase>. "
                    f"{original_question}"
                )
                entries.append({
                    "question": adv_question,
                    "raw": rephrased_raw,
                    "answer": original_answer,
                    "original_question": original_question,
                    "is_adverserial": True,
                })

    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Done. Wrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a JSONL file to a flat JSON list with adversarial entries."
    )
    parser.add_argument("--input", required=True, help="Path to input .jsonl file")
    parser.add_argument("--output", required=True, help="Path to output .json file")
    args = parser.parse_args()

    convert_jsonl_to_json(args.input, args.output)


if __name__ == "__main__":
    main()