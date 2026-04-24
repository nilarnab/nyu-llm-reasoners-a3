import argparse

from student.sec_7.defaults import MODEL_NAME

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # change
    parser.add_argument("--train_dataset_path", type=str, default="student/data/pit/pit-train.jsonl")
    parser.add_argument("--test_dataset_path", type=str, default="student/data/pit/pit-test.jsonl")
    # ==

    parser.add_argument("--reduce", type=float, default=1)
    parser.add_argument("--eval_after", type=int, default=5)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)

    args = parser.parse_args()


