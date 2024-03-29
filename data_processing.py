# run pdf reader
from sys import argv

from app.data_processing.endnote_pdf_reader import pdf_reader
from app.data_processing.preprocess import PreprocessTFIDF


def main(name, ris_path, pdf_path, export_dir_path):
    """
    Main function which calls the pdf reader and preprocess functions
    """
    pdf_processed_file_path = pdf_reader(ris_path, pdf_path, name)
    # run preprocess
    preprocess_obj = PreprocessTFIDF(pdf_processed_file_path, name)
    preprocess_obj.process()

    # export preprocessed file
    preprocess_obj.export(export_dir_path)


# example run of the main function

# TODO: add a function somehow to get the name of the file from shell script


if __name__ == "__main__":
    ris_path = argv[1]
    pdf_path = argv[2]
    output_dir = argv[3]
    ris_path_slited = ris_path.split("/")
    name = ris_path_slited[-1].replace(".ris", "")
    main(
        name=name,
        ris_path=ris_path,
        pdf_path=pdf_path,
        export_dir_path=output_dir,
    )
