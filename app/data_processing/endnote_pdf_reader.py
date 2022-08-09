import pdftotext
import rispy

from app.import_export import export_data


def pdf_reader(ris_path, pdf_path, name):
    """
    Read pdf file and save it to a pickle file
    """

    def read_ris(file_path):
        """
        Reads a RIS file and returns a list of dictionaries.
        """
        with open(file_path, "r") as f:
            entries = rispy.load(f)
        return entries

    def pdftotext_reader(file_name):
        def check_the_file_is_available_or_not(file_name):
            try:
                f = open(file_name, "r")
                f.close()
                return True
            except FileNotFoundError:
                return False

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
    return "asset/endnote_ris/" + name + ".pickle"
