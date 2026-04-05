import pytesseract
import cv2
import re

def extract_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def validate_text(text):
    score = 0

    if re.search(r'[A-Z]{2}[0-9]{4}', text):
        score += 1

    if re.search(r'(0[1-9]|1[0-2])/[0-9]{2}', text):
        score += 1

    if "paracetamol" in text.lower():
        score += 1

    return score / 3