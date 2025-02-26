import pdfplumber
import pdf2image
import pytesseract
import os
from PIL import Image
import numpy as np

def extract_text_from_image(image):
    """
    Extracts text from an image using OCR (Tesseract).
    """
    # Convert the image to grayscale (using PIL, instead of cv2)
    gray_image = image.convert("L")
    np_image = np.array(gray_image)
    _, binary = cv2.threshold(np_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    extracted_text = pytesseract.image_to_string(binary, lang="eng")  # OCR
    return extracted_text.strip()

def extract_pdf_content(pdf_path):
    extracted_text = ""
    tables_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            extracted_text += page_text
            print(f"🔍 Extracted Text from Page: {page_text[:500]}")  # ✅ Debugging text extraction

            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                if table:
                    formatted_table = "\n".join([" | ".join(filter(None, row)) for row in table if any(row)])
                    tables_data.append(formatted_table)

    pdf_images = pdf2image.convert_from_path(pdf_path)
    image_text = ""
    for image in pdf_images:
        image_text += extract_text_from_image(image) + "\n"

    full_content = (
        extracted_text +
        "\n\n[TABLE DATA]\n" + "\n".join(tables_data) +
        "\n\n[IMAGE TEXT]\n" + image_text
    )

    print(f"🔍 Full Extracted Content: {full_content[:500]}")  # ✅ Debugging extracted text

    return full_content.strip()
