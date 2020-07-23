import numpy as np
import os

def main():
    preds_dir = 'preds/'
    files = ['in_words.npy', 'sent_ids.npy', 'sent_lengths.npy', 'pred_ids.npy', 'probs.npy']
    peek_files(preds_dir, files)
    check_min_value(preds_dir, files)

def peek_files(preds_dir, files):
    for file in files:
        with open(os.path.join(preds_dir, file), 'rb') as f:
            row1 = np.load(f)
            row2 = np.load(f)
            print(f"for {file}, row1 is {row1} and it's shape is {row1.shape}")
            print(f"for {file}, row2 is {row2} and it's shape is {row2.shape}")

def check_min_value(preds_dir, files):
    for file in files:
        with open(os.path.join(preds_dir, file), 'rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            print(file)
            while f.tell() < fsz:
                row = np.load(f)
                print(row.min())

if __name__ == "__main__":
    main()