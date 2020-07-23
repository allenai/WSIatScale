import argparse
import numpy as np
import os

def main(args):
    files = ['in_words.npy', 'sent_ids.npy', 'sent_lengths.npy', 'pred_ids.npy', 'probs.npy']
    for file in files:
        with open(os.path.join(args.preds_dir, file), 'rb') as f:
            row1 = np.load(f)
            row2 = np.load(f)
            print(f"for {file}, row1 is {row1} and it's shape is {row1.shape}")
            print(f"for {file}, row2 is {row2} and it's shape is {row2.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_dir", type=str, required=True)
    args = parser.parse_args()

    main(args)