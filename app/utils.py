import os
import re
import numpy as np
from transformers import AutoModel, PhobertTokenizer
from nltk import word_tokenize
import torch
import py_vncorenlp
from py_vncorenlp import VnCoreNLP
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"

# Load VnCoreNLP models
# Đường dẫn tương đối đến VnCoreNLP (nằm trong cùng thư mục với app.py)
VNCORP_PATH = os.path.join(os.path.dirname(__file__), "VnCoreNLP-master")

# Tải model (nếu cần)
py_vncorenlp.download_model(save_dir=VNCORP_PATH)

# Load model
model = py_vncorenlp.VnCoreNLP(save_dir=VNCORP_PATH)

def load_bert():
    v_phobert = AutoModel.from_pretrained('vinai/phobert-base')
    v_tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

def standardize_data(row):
    if not isinstance(row, str):
        text = str(row)

    # Define the pattern to match unwanted characters
    pattern = r'[^a-zA-ZÀ-Ỹà-ỹ\s]'  # Keep letters, digits, spaces, and Vietnamese characters

    # Use regex to substitute unwanted characters with an empty string
    cleaned_text = re.sub(pattern, ' ', row)

    return str(cleaned_text.lower())

def remove_repetitive_characters(text):
    # Define regex pattern to match repetitive characters
    pattern = re.sub(r'(.)\1+', r'\1\1', text)  # Match one character followed by one or more occurrences of the same character
    return pattern

def correct_spelling_teencode(text):
    # Dictionary of common teencode and their correct forms
    teencode_dict = {
        'chx': 'chưa',
        'z': 'vậy',
        'd': 'vậy',
        'k': 'không',
        'hok': 'không',
        'ko': 'không',
        'kh': 'không',
        'cx': 'cũng',
        'đỉm': 'điểm',
        'oce': 'ok',
        'oke': 'ok',
        'đc': 'được',
        'ns': 'nói',
        'tc': 'tính chất',
        'tch': 'tính chất',
        'tks': 'cảm ơn',
        'nc': 'nói chuyện',
        'thui': 'thôi',
        'ha': 'hình ảnh',
        'ik': 'đi',
        'auce': 'ok',
        'xink': 'xinh',
        'dth': 'dễ thương',
        'dthw': 'dễ thương',
        'nhe':'nha',
        'nthe': 'như thế',
        'dethun': 'dễ thương',
        'kcj': 'không có gì',
        'kcgi': 'không có gì',
        'ntn': 'như thế này',
        'ng': 'người',
        'mn': 'mọi người',
        'ng': 'mọi người',
        'nma': 'nhưng mà',
        'qlai': 'quay lại',
        'sp': 'sản phẩm',
        'tn': 'tin nhắn',
        'qtam': 'quan tâm',
        'th': 'thôi',
        'nch': 'nói chung',
        'mk': 'mình'

         # Add more teencode mappings as needed
    }

    # Tokenize the text into words
    words = word_tokenize(text)

    # Replace teencode with correct forms
    corrected_words = [teencode_dict[word] if word in teencode_dict else word for word in words]

    # Join the corrected words back into a single string
    corrected_text = ' '.join(corrected_words)
    return corrected_text

# Hàm tiền xử lý văn bản (giả sử)
def preprocess_text(text: str) -> str:
    text = standardize_data(text)  # Chuẩn hóa dữ liệu đầu vào
    text = correct_spelling_teencode(text)  # Sửa lỗi chính tả teencode
    text = remove_repetitive_characters(text)  # Xóa các ký tự lặp lại
    return text if text.strip() != "" else None  # Trả về None nếu text rỗng


def extract_phobert_features_single(text, max_len=50):
    """
    Process a single text through preprocessing, PhoBERT tokenization, and feature extraction
    
    Parameters:
    -----------
    text : str
        Input text to process
    max_len : int, default=50
        Maximum sequence length for tokenization
        
    Returns:
    --------
    numpy.ndarray
        Feature vector for the input text
    """
    # Load BERT model and tokenizer
    v_phobert, v_tokenizer = load_bert()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    v_phobert = v_phobert.to(device)
    
    # Word segmentation
    segmented_text = model.word_segment(text)
    processed_text = ' '.join(segmented_text)
    
    # Tokenize
    encoding = v_tokenizer(
        processed_text,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Extract features
    with torch.no_grad():
        outputs = v_phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Get the [CLS] token embeddings
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    # Clear GPU memory if using CUDA
    if device == 'cuda':
        del input_ids
        del attention_mask
        torch.cuda.empty_cache()
    
    return features[0]  # Return the single feature vector