# import pdfplumber
# import pdf2image
# import pytesseract
# import os
# from PIL import Image
# import cv2
# import numpy as np

# def extract_text_from_image(image):
#     """
#     Extracts text from an image using OCR (Tesseract).
#     """
#     gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     extracted_text = pytesseract.image_to_string(binary, lang="eng")  # OCR
#     return extracted_text.strip()

# def extract_pdf_content(pdf_path):
#     """
#     Extracts text, tables, and image-based text from a PDF without specifying Poppler.
#     """
#     extracted_text = ""
#     tables_data = []

#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             extracted_text += page.extract_text() or ""

#             # ✅ Extract tables
#             tables = page.extract_tables()
#             for table in tables:
#                 if table:
#                     formatted_table = "\n".join([" | ".join(filter(None, row)) for row in table if any(row)])
#                     tables_data.append(formatted_table)

#     # ✅ Extract images and process OCR (Without specifying Poppler)
#     try:
#         pdf_images = pdf2image.convert_from_path(pdf_path)  # ❌ No poppler_path needed
#         image_text = ""
#         for image in pdf_images:
#             image_text += extract_text_from_image(image) + "\n"
#     except Exception as e:
#         image_text = f"Error extracting images: {str(e)}"

#     # ✅ Combine extracted content
#     full_extracted_content = (
#         extracted_text +
#         "\n\n[TABLE DATA]\n" + "\n".join(tables_data) + 
#         "\n\n[IMAGE TEXT]\n" + image_text
#     )

#     return full_extracted_content.strip()
