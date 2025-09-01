import pdfplumber
import pytesseract
from PIL import Image
import PyPDF2

# Path to Tesseract (update if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Shaikh Irfan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    - Uses PyPDF2 first (for selectable text).
    - Falls back to OCR with Tesseract for scanned pages.
    """
    text = ""

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            print(f"PDF has {len(reader.pages)} pages.")

            # Process each page
            for i, page in enumerate(reader.pages):
                page_text = ""
                try:
                    # --- Try PyPDF2 ---
                    page_text = page.extract_text()
                except Exception as e:
                    print(f"PyPDF2 failed on page {i}: {e}")

                # If no text, do OCR
                if not page_text or not page_text.strip():
                    try:
                        with pdfplumber.open(pdf_path) as pdf:
                            pil_img = pdf.pages[i].to_image(resolution=300).original
                            page_text = pytesseract.image_to_string(pil_img)
                    except Exception as e:
                        print(f"OCR failed on page {i}: {e}")
                        page_text = ""

                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"Failed to open PDF: {e}")

    return text.strip()
