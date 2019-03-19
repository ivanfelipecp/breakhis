import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model")
parser.add_argument("--epochs")
parser.add_argument("--mag")
parser.add_argument("--dataset")
parser.add_argument("--batch_size")
parser.add_argument("--lr")
parser.add_argument("--binary")
args = parser.parse_args()

print(args)