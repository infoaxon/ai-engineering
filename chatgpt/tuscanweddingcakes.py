import fitz  # PyMuPDF

doc = fitz.open("/Users/shubhamnagar/Downloads/Report.pdf")
for page in doc:
    text_instances = page.search_for("06/07/2023")
    for inst in text_instances:
        page.add_redact_annot(inst, text="06/07/2026")
    page.apply_redactions()

doc.save("updated.pdf")
