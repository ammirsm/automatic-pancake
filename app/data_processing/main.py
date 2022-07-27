# run pdf reader
from app.data_processing.endnote_pdf_reader import pdf_reader
from app.data_processing.preprocess import PreprocessTFIDF


def main(name, ris_path, pdf_path, export_file_path):
    """
    Main function which calls the pdf reader and preprocess functions
    """
    pdf_processed_file_path = pdf_reader(ris_path, pdf_path, name)
    # run preprocess
    preprocess_obj = PreprocessTFIDF(pdf_processed_file_path)
    preprocess_obj.process()

    # export preprocessed file
    preprocess_obj.export(export_file_path)


# example run of the main function

# TODO: add a function somehow to get the name of the file from shell script

name = "vande"
ris_path = "./asset/endnote_ris/" + name + ".ris"
pdf_path = (
    "/Users/amirhossein/Documents/Meta-analysis/van de schoot 2017/vande.Data/PDF/"
)
export_file_path = "./asset/endnote_ris/" + name + ".pickle"

main(name, ris_path, pdf_path, export_file_path)
