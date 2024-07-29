import os
import gc
import torch
import argparse
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from pdf2image import convert_from_path
from tqdm import tqdm

# Function to clear the TXT output file if it exists
def clear_txt_output(file_path):
    with open(file_path, 'w') as f:
        f.write('')

# Function to convert PDF to images
def pdf_to_img(pdf_path):
    images = convert_from_path(pdf_path, fmt='jpeg')
    return images

# Load the table detection model and image processor
model_name = "microsoft/table-transformer-detection"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = TableTransformerForObjectDetection.from_pretrained(model_name, revision="no_timm")

# Function to detect tables in a batch of images
def detect_table(image_doc):
    results = []

    for image in image_doc:
        # Preprocess image document
        inputs = image_processor(images=image, return_tensors="pt")

        # Detect tables
        outputs = model(**inputs)

        # Convert outputs to Pascal VOC format
        target_sizes = torch.tensor([image.size[::-1]])
        page_results = image_processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)

        results.append({"image": image, "Result": page_results})

    return results

# Function to extract bounding boxes for detected tables
def get_table_bbox(results, label_map, lower_threshold, upper_threshold):
    tables_coordinates = []

    for page_idx, page_results in enumerate(results):
        for detection in page_results["Result"]:
            boxes = detection["boxes"]
            scores = detection["scores"]
            labels = detection["labels"]

            for score, label, box in zip(scores, labels, boxes):
                if lower_threshold <= score <= upper_threshold:
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

# Function to process a batch of pages and extract table details
def process_batch(batch_images, start_page, lower_threshold, upper_threshold):
    results = []
    table_results = detect_table(batch_images)
    label_map = {1: 'table'}
    table_bbox = get_table_bbox(table_results, label_map, lower_threshold, upper_threshold)

    for bbox in table_bbox:
        bbox["page"] = start_page + bbox["page"]  # Adjust page number by start_page
        results.append(bbox)

    return results

def process_pdf(pdf_path, lower_threshold, output_txt_file, batch_size=5, upper_threshold=1.0):
    # Clear the TXT output file
    clear_txt_output(output_txt_file)

    # Main processing
    images = pdf_to_img(pdf_path)
    total_pages = len(images)
    num_batches = (total_pages + batch_size - 1) // batch_size  # Calculate number of batches

    all_results = []

    for batch_num in tqdm(range(num_batches), desc="Processing batches"):
        start_page = batch_num * batch_size
        end_page = min((batch_num + 1) * batch_size, total_pages)
        batch_images = images[start_page:end_page]

        # Process the current batch
        batch_results = process_batch(batch_images, start_page, lower_threshold, upper_threshold)
        all_results.extend(batch_results)

        # Clear variables and perform garbage collection to manage memory
        del batch_images
        gc.collect()
        torch.cuda.empty_cache()  # If using a GPU, free up GPU memory

    # Save results to TXT
    with open(output_txt_file, 'w') as f:
        for result in all_results:
            f.write(f"Page: {result['page']}, Bounding Box: ({result['xmin']}, {result['ymin']}, {result['xmax']}, {result['ymax']}), "
                    f"Label: {result['label']}, Confidence: {result['confidence']}\n")

    print(f"All tables extracted and saved to {output_txt_file}")

def main(input_dir, lower_threshold, output_dir, batch_size=5, upper_threshold=1.0):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                relative_path = os.path.relpath(pdf_path, input_dir)
                output_txt_path = os.path.join(output_dir, relative_path.replace('.pdf', '.txt'))

                os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

                print(f"Processing PDF: {pdf_path}")
                process_pdf(pdf_path, lower_threshold, output_txt_path, batch_size, upper_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tables from all PDF files in a directory and save to a specified output directory.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing PDF files.")
    parser.add_argument("lower_threshold", type=float, help="Lower confidence threshold for table detection (e.g., 0.9).")
    parser.add_argument("output_dir", type=str, help="Path to the output directory to save extracted tables.")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of pages to process in a batch. Default is 5.")
    parser.add_argument("--upper_threshold", type=float, default=1.0, help="Upper confidence threshold for table detection. Default is 1.0.")

    args = parser.parse_args()

    main(args.input_dir, args.lower_threshold, args.output_dir, args.batch_size, args.upper_threshold)
