# Pdf-Table_Extraction-

Problem Statement:-
To develop an automated solution for accurately detecting and extracting tables from PDF documents.
This involves identifying table structures, extracting cell content, and mapping word boundin
boxes to improve the accuracy and consistency of table extraction.


Method : Used "Table-Transformer-detection" and "Table-Transformer-Structure" models for detecting
and analysing tables in images.

Analysis: Bounding boxes and scores were extracted correctly but inconsistent for complex tables.


# Table-Detetion-Script -
This Python script utilizes the Table Transformer model for accurate table detection within
documents. It extracts bounding box values of detected tables, along with their associated
confidence scores. 

Dependencies:
Install via pip:
pip install torch transformers pdf2image tqdm
The model is auto-downloaded when first run.

Usage:
Run with:
python extract_tables.py <pdf_path> <threshold> <output_txt_file> [--batch_size BATCH_SIZE] [--upper_threshold UPPER_THRESHOLD]

<pdf_path>: Input PDF file.
<threshold>: Confidence threshold (e.g., 0.9).
<output_txt_file>: Output TXT file.
[--batch_size BATCH_SIZE]: (Optional) Pages per batch (default: 5).
[--upper_threshold UPPER_THRESHOLD]: (Optional) Upper confidence threshold (default: 1.0).
Example:
python extract_tables.py C:\path\to\your.pdf 0.9 C:\path\to\output.txt --batch_size 10 --upper_threshold 0.95

Help:
python extract_tables.py --help

Summary:

Install dependencies:
pip install torch transformers pdf2image tqdm
Run the script:
python extract_tables.py <pdf_path> <threshold> <output_txt_file> [--batch_size BATCH_SIZE] [--upper_threshold UPPER_THRESHOLD]
For help:
python extract_tables.py --help
