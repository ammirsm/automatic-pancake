import pdftotext
import rispy

from app.import_export import export_data

name = "vande"

ris_path = "./asset/endnote_ris/" + name + ".ris"
pdf_path = (
    "/Users/amirhossein/Documents/Meta-analysis/van de schoot 2017/vande.Data/PDF/"
)


def read_ris(file_path):
    """
    Reads a RIS file and returns a list of dictionaries.
    """
    with open(file_path, "r") as f:
        entries = rispy.load(f)
    return entries


def check_the_file_is_available_or_not(file_name):
    try:
        f = open(file_name, "r")
        f.close()
        return True
    except FileNotFoundError:
        return False


def pdftotext_reader(file_name):
    print(file_name)
    if check_the_file_is_available_or_not(file_name):
        with open(file_name, "rb") as f:
            pdf = pdftotext.PDF(f)
        return " ".join(pdf)
    return ""


papers = read_ris(ris_path)
for i, entry in enumerate(papers):
    if "file_attachments1" in entry:
        papers[i]["fulltext"] = pdftotext_reader(
            pdf_path + entry["file_attachments1"].split("//")[1]
        )


export_data(papers, "asset/endnote_ris/" + name + ".pickle")
