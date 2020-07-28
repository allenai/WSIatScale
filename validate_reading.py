import numpy as np
import os

def main():
    preds_dir = 'preds/'
    files = ['in_words.npy', 'sent_ids.npy', 'sent_lengths.npy', 'pred_ids.npy', 'probs.npy']
    peek_files(preds_dir, files)
    check_stats(preds_dir, files)

def peek_files(preds_dir, files):
    for file in files:
        with open(os.path.join(preds_dir, file), 'rb') as f:
            row1 = np.load(f)
            row2 = np.load(f)
            print(f"for {file}, row1 is {row1} and it's shape is {row1.shape}\n")
            print(f"for {file}, row2 is {row2} and it's shape is {row2.shape}\n")

def check_stats(preds_dir, files):
    for file in files:
        min_over_rows = 99999
        num_rows = 0
        with open(os.path.join(preds_dir, file), 'rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            while f.tell() < fsz:
                num_rows += 1
                row = np.load(f)
                min_over_rows = min(min_over_rows, row.min())
            print(f"{file} min value: {min_over_rows}")
            print(f"{file} num rows: {num_rows}")

if __name__ == "__main__":
    main()