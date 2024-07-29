import os
import gc
import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from pdf2image import convert_from_path
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def pdf_to_img(pdf_path):
    images = convert_from_path(pdf_path, fmt='jpeg')
    return images


def detect_table(image_doc, image_processor, threshold):
    results = []
    for image in image_doc:
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        page_results = image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)
        results.append({"image": image, "Result": page_results})
    return results


def get_table_bbox(results, label_map, max_threshold):
    tables_coordinates = []
    for page_idx, page_results in enumerate(results):
        for detection in page_results["Result"]:
            boxes = detection["boxes"]
            scores = detection["scores"]
            labels = detection["labels"]
            for score, label, box in zip(scores, labels, boxes):
                if score >= max_threshold:
                    xmin, ymin, xmax, ymax = [round(i, 2) for i in box.tolist()]
                    table_dict = {
                        "page": page_idx + 1,
                        "label": label_map.get(label.item(), "Unknown"),
                        "confidence": round(score.item(), 3),
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax
                    }
                    tables_coordinates.append(table_dict)
    return tables_coordinates


def highlight_tables(image, table_boxes, padding):
    doc_image = image.copy()
    draw = ImageDraw.Draw(doc_image)
    for table in table_boxes:
        rectangle_coords = (table["xmin"] - padding,
                            table["ymin"] - padding,
                            table["xmax"] + padding,
                            table["ymax"] + padding)
        draw.rectangle(rectangle_coords, outline="red", width=2)
    return doc_image


def get_cropped_images(image, table_bbox, padding):
    cropped_images = []
    if not table_bbox:
        return cropped_images
    for table in table_bbox:
        left = max(table["xmin"] - padding, 0)
        top = max(table["ymin"] - padding, 0)
        right = min(table["xmax"] + padding, image.width)
        bottom = min(table["ymax"] + padding, image.height)
        cropped_image = image.crop((left, top, right, bottom))
        cropped_images.append(cropped_image)
    return cropped_images


def get_table_features(cropped_image):
    inputs = image_processor(images=cropped_image, return_tensors="pt")
    with torch.no_grad():
        outputs = structure_model(**inputs)
    target_sizes = torch.tensor([cropped_image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    features = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(coord, 2) for coord in box.tolist()]
        score = score.item()
        label = structure_model.config.id2label[label.item()]
        features.append({"label": label, "score": score, "bbox": box})
    return features


def display_detected_features(cropped_image, features):
    cropped_table_visualized = cropped_image.copy()
    draw = ImageDraw.Draw(cropped_table_visualized)
    font_path = "Roboto-Regular.ttf"
    try:
        font = ImageFont.truetype(font_path, 15)
    except IOError:
        font = ImageFont.load_default()
    for feature in features:
        draw.rectangle(feature["bbox"], outline="red", width=2)
        text_position = (feature["bbox"][0], max(feature["bbox"][1] - 15, 0))
        draw.text(text_position, feature["label"], fill="blue", font=font)
    return cropped_table_visualized


def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])
    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
            row_cells.append({'cell': cell_bbox})
        cell_coordinates.append({'cells': row_cells, 'cell_count': len(row_cells)})
    return cell_coordinates


def apply_ocr(cell_coordinates, cropped_image):
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(cell_coordinates):
        row_text = []
        for cell in row["cells"]:
            cell_image = np.array(cropped_image.crop(cell["cell"]))
            text = pytesseract.image_to_string(cell_image, lang='eng', config='--psm 6').strip()
            if text:
                row_text.append(text)
        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)
        data[idx] = row_text
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data
    return data


def process_batch(batch_images, batch_num, threshold, max_threshold):
    table_results = detect_table(batch_images, image_processor, threshold)
    label_map = {1: 'table'}
    table_bbox = get_table_bbox(table_results, label_map, max_threshold)
    padding = 40
    all_cropped_images = []
    for i, image in enumerate(tqdm(batch_images, desc=f"Processing Batch {batch_num}, Page", leave=False)):
        page_num = i + 1
        table_boxes = [table for table in table_bbox if table["page"] == page_num]
        cropped_images = get_cropped_images(image, table_boxes, padding)
        all_cropped_images.extend(cropped_images)
    tables_features = {}
    for idx, cropped_image in enumerate(all_cropped_images):
        features = get_table_features(cropped_image)
        tables_features[idx] = {"image": cropped_image, "features": features}
    all_table_coordinates = []
    for table_id, data in tables_features.items():
        table_coordinates = get_cell_coordinates_by_row(data["features"])
        all_table_coordinates.append({"ID": table_id, "Coordinates": table_coordinates})
    for table_coordinate in tqdm(all_table_coordinates, desc="Applying OCR", leave=False):
        table_data = apply_ocr(table_coordinate['Coordinates'], all_cropped_images[table_coordinate['ID']])
        all_table_data.append({
            "Batch": batch_num,
            "TableID": table_coordinate['ID'],
            "TableData": dataframe_to_json(table_data)
        })


def dataframe_to_json(table_data):
    df = pd.DataFrame(table_data)
    return df.to_dict(orient='records')


def main():
    parser = argparse.ArgumentParser(description="Extract tables from PDF files and save as JSON.")
    parser.add_argument("input_dir", type=str, help="Directory containing input PDF files.")
    parser.add_argument("output_dir", type=str, help="Directory to save extracted JSON files.")
    parser.add_argument("threshold", type=float, help="Threshold value for table detection (0 to 1).")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of pages to process in each batch (default is 5).")
    parser.add_argument("--max_threshold", type=float, default=1.00, help="Maximum threshold value for table detection (default is 0.96).")
    args = parser.parse_args()
    global image_processor, model, structure_model, all_table_data
    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
    structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition", revision="no_timm")
    all_table_data = []

    input_files = [f for f in os.listdir(args.input_dir) if f.endswith('.pdf')]
    total_pdfs = len(input_files)

    with tqdm(total=total_pdfs, desc="Processing PDFs", leave=True) as pbar_pdfs:
        for file in input_files:
            pbar_pdfs.set_postfix(file=file)
            pdf_path = os.path.join(args.input_dir, file)
            batch_images = pdf_to_img(pdf_path)
            batch_num = 1

            with tqdm(total=len(batch_images), desc=f"Processing {file}", leave=True) as pbar_pdf:
                for i in range(0, len(batch_images), args.batch_size):
                    batch = batch_images[i:i + args.batch_size]
                    process_batch(batch, batch_num, args.threshold, args.max_threshold)
                    batch_num += 1
                    pbar_pdf.update(len(batch))

            pbar_pdfs.update(1)
    output_file = os.path.join(args.output_dir, "extracted_tables.json")
    with open(output_file, "w") as f:
        json.dump(all_table_data, f, indent=4)
    gc.collect()

if __name__ == "__main__":
    main()
