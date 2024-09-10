import os
import re
import nltk
import fitz
import pandas as pd
import csv

# Ensure NLTK tokenizers are available
nltk.download('punkt')


def read_pdfs_in_directory(directory_path, extract_text_from_pdf):
    pdf_texts = []

    # Traverse through all files and subdirectories in the given directory
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                with open(pdf_path, 'rb') as pdf_file:
                    # Use the provided function to extract text from the PDF
                    text = extract_text_from_pdf(pdf_file)
                    pdf_texts.append(text)

    return pdf_texts


def split_text_into_sentences(text, max_tokens=500):
    sentences = nltk.sent_tokenize(text)
    result = []
    current_chunk = []
    current_chunk_len = 0

    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        if current_chunk_len + len(tokens) > max_tokens:
            result.append(' '.join(current_chunk))
            current_chunk = []
            current_chunk_len = 0

        current_chunk.append(sentence)
        current_chunk_len += len(tokens)

    if current_chunk:
        result.append(' '.join(current_chunk))

    return result


def extract_text_from_images(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()

    doc.close()
    return text


def clean_text(text):
    # Remove special characters, periods, and extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\.', '', text)  # Remove periods
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing spaces
    return text





directory_path = '/mnt/c/Users/Administrateur/Downloads/PFE livres/PFE livres/'

pdf_texts = read_pdfs_in_directory(directory_path, extract_text_from_images)
print(pdf_texts)
split_pdf_texts = [split_text_into_sentences(text) for text in pdf_texts]
flattened_split_pdf_texts = [sentence for sublist in split_pdf_texts for sentence in sublist]



def save_list_to_text_file(file_path, text_list, separator="*************************************************************"):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in text_list:
            file.write(f"{item}{separator}")

# Example usage
save_list_to_text_file('pdf_texts.txt', flattened_split_pdf_texts)

