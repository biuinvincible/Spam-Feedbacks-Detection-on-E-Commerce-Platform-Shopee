import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from utils import (
    remove_repetitive_characters, 
    correct_spelling_teencode, 
    standardize_data,
    extract_phobert_features
)

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text)

def preprocess_data(input_path, output_dir, model_type="xgboost"):
    df = pd.read_csv(input_path)

    # Convert sentiment to binary labels
    df["label"] = df["sentiment"].apply(lambda x: 1 if x == "Spam" else 0)

    # Basic preprocessing
    df["processed_comment"] = df["comment"].apply(lambda x: standardize_data(x))
    df["processed_comment"] = df["processed_comment"].apply(lambda x: correct_spelling_teencode(x))
    df["processed_comment"] = df["processed_comment"].apply(lambda x: remove_repetitive_characters(x))
    df = df[df["processed_comment"] != ""]

    # Xử lý riêng cho XGBoost
    if model_type == "xgboost":
        # Sử dụng PhoBERT features
        X = extract_phobert_features(df, "processed_comment")
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Giữ nguyên xử lý cho các model deep learning
    elif model_type in ["lstm", "gru", "cnn"]:
        import nltk
        nltk.download('punkt')
        tokenizer = get_tokenizer('basic_english')

        # Xây dựng từ điển
        vocab = build_vocab_from_iterator(
            yield_tokens(df["processed_comment"], tokenizer),
            min_freq=2,
            specials=["<unk>", "<pad>"]
        )
        vocab.set_default_index(vocab["<unk>"])

        # Lưu từ điển
        os.makedirs(output_dir, exist_ok=True)
        torch.save(vocab, os.path.join(output_dir, f"{model_type}_vocab.pth"))

        # Tokenize và chuyển đổi văn bản thành chỉ số từ điển
        max_len = 50  # Độ dài tối đa của sequence

        def text_pipeline(text):
            tokens = tokenizer(text)
            indices = [vocab[token] for token in tokens]
            if len(indices) < max_len:
                indices = indices + [vocab["<pad>"]] * (max_len - len(indices))
            else:
                indices = indices[:max_len]
            return indices

        # Chuyển đổi văn bản thành số
        X = np.array([text_pipeline(text) for text in df["processed_comment"]])
        y = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Lưu thông tin về kích thước từ điển
        vocab_info = {
            "vocab_size": len(vocab),
            "max_len": max_len,
            "pad_idx": vocab["<pad>"]
        }
        np.save(os.path.join(output_dir, f"{model_type}_vocab_info.npy"), vocab_info)

    # Lưu dữ liệu đã xử lý
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{model_type}_X_train.npy"), X_train)
    np.save(os.path.join(output_dir, f"{model_type}_X_test.npy"), X_test)
    np.save(os.path.join(output_dir, f"{model_type}_y_train.npy"), y_train)
    np.save(os.path.join(output_dir, f"{model_type}_y_test.npy"), y_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Lấy đường dẫn thư mục chứa preprocessing.py
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    base_data_path = os.path.join(BASE_DIR, "data", "raw", "labeled_data.csv")
    output_data_path = os.path.join(BASE_DIR, "data", "processed")

    # Xử lý cho tất cả các loại model
    preprocess_data(base_data_path, output_data_path, model_type="xgboost")
    preprocess_data(base_data_path, output_data_path, model_type="lstm")
    preprocess_data(base_data_path, output_data_path, model_type="gru")
    preprocess_data(base_data_path, output_data_path, model_type="cnn")