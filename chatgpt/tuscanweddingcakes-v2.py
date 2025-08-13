import fitz  # PyMuPDF

doc = fitz.open("/Users/shubhamnagar/Downloads/Report.pdf")
for page in doc:
    for inst in page.search_for("06/07/2023"):
        page.insert_text(inst[:2], "06/07/2026", fontname="helv", fontsize=10, color=(0, 0, 0))

doc.save("updated-v2.pdf")
