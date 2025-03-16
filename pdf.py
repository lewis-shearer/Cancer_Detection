import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import pathlib

# Register Arial and Arial-Bold
pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'arialbd.ttf'))

# Sample data (replace with your actual data)
patient_data = {
    "patient_id": "P12345",
    "patient_name": "John Doe",
    "date_of_birth": "1970-01-15",
    "gender": "Male",
}

mri_data = {
    "date_of_scan": "2023-11-15",
    "scan_type": "T1-weighted",
    "area_of_scan": "Brain",
    "image_path": "images/glimoa.jpg",
}

cnn_results = {
    "prediction": "Malignant Tumor Detected",
    "probability": 0.85,
}

accuracy_data = {
    "overall_accuracy": 0.92,
    "sensitivity": 0.88,
    "specificity": 0.95,
}

# Define report data sections
patient_section = [
    ["Patient ID:", patient_data["patient_id"]],
    ["Patient Name:", patient_data["patient_name"]],
    ["Date of Birth:", patient_data["date_of_birth"]],
    ["Gender:", patient_data["gender"]],
]

mri_section = [
    ["Date of Scan:", mri_data["date_of_scan"]],
    ["Scan Type:", mri_data["scan_type"]],
    ["Area of Scan:", mri_data["area_of_scan"]],
]

cnn_section = [
    ["Prediction:", cnn_results["prediction"]],
    ["Probability:", f"{cnn_results['probability']:.2%}"],
]

accuracy_section = [
    ["Overall Accuracy:", f"{accuracy_data['overall_accuracy']:.2%}"],
    ["Sensitivity:", f"{accuracy_data['sensitivity']:.2%}"],
    ["Specificity:", f"{accuracy_data['specificity']:.2%}"],
]

# Create tables for each section
def create_styled_table(data, title):
    table = Table(data)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.aliceblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.darkslategray),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Arial'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('BOX', (0, 0), (-1, -1), 0, colors.white),
        ('FONTNAME', (0, 0), (0, -1), 'Arial-Bold'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ])
    table.setStyle(style)

    styles = getSampleStyleSheet()
    title_paragraph = Paragraph(title, styles['Heading2'])
    title_paragraph.style.fontName = 'Arial-Bold'
    return [title_paragraph, Spacer(1, 12), table, Spacer(1, 18)]

# Get the Downloads folder path
downloads_path = str(pathlib.Path.home() / "Downloads")
output_pdf_path = os.path.join(downloads_path, "medical_report_glioma.pdf")

# Create a PDF document
doc = SimpleDocTemplate(output_pdf_path, pagesize=letter,
                        leftMargin=inch, rightMargin=inch,
                        topMargin=inch, bottomMargin=inch)

elements = []

styles = getSampleStyleSheet()
title = Paragraph("CNN Cancer Detection Analysis Report", styles['Heading1'])
title.style.fontName = 'Arial-Bold'
title.style.textColor = colors.royalblue
elements.append(title)
elements.append(Spacer(1, 24))

elements.extend(create_styled_table(patient_section, "Patient Information"))

# Add MRI Image and Details side-by-side
if mri_data["image_path"] and os.path.exists(mri_data["image_path"]):
    try:
        img = Image(mri_data["image_path"], width=3 * inch, height=3 * inch, hAlign='LEFT')
        table_mri = create_styled_table(mri_section, "MRI Scan Details")[2]

        side_by_side_table = Table([[img, table_mri]], colWidths=[3 * inch, 4 * inch])
        side_by_side_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))

        elements.extend([Paragraph("MRI Scan:", styles['Heading2']), Spacer(1, 6), side_by_side_table, Spacer(1, 18)])

    except Exception as e:
        print(f"Error loading image: {e}")
else:
    print(f"Image not found at: {mri_data['image_path']}")

elements.extend(create_styled_table(cnn_section, "CNN Analysis and Prediction"))
elements.extend(create_styled_table(accuracy_section, "Accuracy Metrics"))

doc.build(elements)

print(f"Medical report generated successfully and saved to: {output_pdf_path}")