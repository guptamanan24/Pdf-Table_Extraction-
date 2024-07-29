
# Automated Table Detection and Extraction from PDF Documents

## Problem Statement
To develop an automated solution for accurately detecting and extracting tables from PDF documents. This involves identifying table structures, extracting cell content, and mapping word bounding boxes to improve the accuracy and consistency of table extraction.

## Method
Used "Table-Transformer-detection" and "Table-Transformer-Structure" models for detecting and analyzing tables in images.

## Analysis
Bounding boxes and scores were extracted correctly but inconsistent for complex tables.

---

# Table-Detection-Script 

This Python script utilizes the Table Transformer model for accurate table detection within documents. It extracts bounding box values of detected tables, along with their associated confidence scores.

### Dependencies
Install via pip:

```sh
pip install torch transformers pdf2image tqdm
```

The model is auto-downloaded when first run.

### Usage
Run with:

```sh
python extract_tables.py <pdf_path> <threshold> <output_txt_file> [--batch_size BATCH_SIZE] [--upper_threshold UPPER_THRESHOLD]
```

- `<pdf_path>`: Input PDF file.
- `<threshold>`: Confidence threshold (e.g., 0.9).
- `<output_txt_file>`: Output TXT file.
- `[--batch_size BATCH_SIZE]`: (Optional) Pages per batch (default: 5).
- `[--upper_threshold UPPER_THRESHOLD]`: (Optional) Upper confidence threshold (default: 1.0).

### Example

```sh
python extract_tables.py C:\path\to\your.pdf 0.9 C:\path\to\output.txt --batch_size 10 --upper_threshold 0.95
```

### Help

```sh
python extract_tables.py --help
```

For more help, you can refer to the documentation for table-detection present in the GitHub repository.

---

# Table-Extraction 

This code extracts tables from PDFs using OCR and Tesseract. It converts PDF pages to images, detects tables with a Table Transformer model, and highlights and crops these tables. The script then applies OCR to extract text from table cells and processes multiple PDFs in batches, saving the structured data in JSON format. It includes memory management to handle large documents efficiently.

### In the folder of different Approaches for Table Detection
The code for the table extraction using OCR is present in the `.pynb` format.

---

# Table-Extraction-Script 

### Description
`Complete_Table_Extraction.py` extracts tables from PDF files, identifies and highlights table structures, and performs OCR to extract text from table cells. It processes pages in batches and saves the extracted data in JSON format.

### Usage

```sh
python Complete_Table_Extraction.py <input_dir> <output_dir> <threshold> [--batch_size <batch_size>] [--max_threshold <max_threshold>]
```

- `<input_dir>`: Directory containing input PDF files.
- `<output_dir>`: Directory to save extracted JSON files.
- `<threshold>`: Threshold value for table detection (0 to 1).
- `--batch_size` (optional): Number of pages per batch (default: 5).
- `--max_threshold` (optional): Maximum threshold value for table detection (default: 1.00).

### Example

```sh
python extract_tables.py ./pdfs ./output 0.9 --batch_size 10 --max_threshold 0.96
```

This command processes PDF files from the `./pdfs` directory, saves the results to the `./output` directory, with a detection threshold of 0.9, a batch size of 10 pages, and a maximum threshold of 0.96 for table detection.

For more help, you can refer to the `Documentation_Extract_Tables` which contains the complete description for running the files.

---

# Improvement in Table Extraction using the Pdftotext

## APPROACH 1
This method combines OCR results from Tesseract with Table Transformer data to accurately map words to bounding boxes.

### Extract Tables Using OCR

1. Perform OCR on table images to get text and bounding boxes for each word.

### Extract and Compare Words

1. Extract words from Table Transformer/OCR and Pdftotext.
2. Compare words from Table Transformer cells with Pdftotext words to find matches.

### Finalize and Scale

1. Finalize matched cells and scale Pdftotext bounding boxes to align with Table Transformer cells.

### Map Words

1. Map Pdftotext words to the adjusted Table Transformer cell bounding boxes.

#### Issues
May incorrectly match words outside the table boundaries in the pdftotext.

### Code
Code is available in the `Different_Approaches_Table_Extraction` folder with file name `Table_Extraction_using_OCR+Pdftotext_Cellwise.pynb`.

---

## APPROACH 2 (CURRENTLY IN PROGRESS)

### Bounding Box Extraction

1. Implemented functions to find bounding boxes for words in Pdftotext using page number and bounding box values of the lines.
2. Developed a function to extract bounding boxes of words from OCR results based on table number and row number.

### Combine Bounding Boxes

1. Combine bounding boxes for each word in Pdftotext lines using their Ymin and Ymax values.

### Match Rows

1. Use the combined bounding boxes to identify and match rows from the Table Transformer/OCR output with corresponding lines from Pdftotext based on the number of matching words.

### Function Creation

1. Developed a function to identify 100% match rows and lines, including table number, page number, row number, and bounding box values.
2. Created a function to extract unique words from these matched rows for each table.



### Limitations

- **OCR Accuracy**: The bounding boxes and word extraction from Tesseract are not yet satisfactory, affecting the overall accuracy of the approach.

### Current Status

The approach is still under development and not yet completed.

### Code

Code is available in the `Different_Approaches_Table_Extraction` folder with file name `Table_Extraction_using_OCR+Pdftotext_rowwise.pynb`.
```
