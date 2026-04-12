from datasets import load_from_disk

dataset = load_from_disk("student/data/countdown/dataset/train")
for i in range(1):
    print(dataset[i])