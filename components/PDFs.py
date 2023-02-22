from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from io import StringIO
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
import glob
from PyPDF2 import PdfFileWriter, PdfFileReader
import os
from tabula import read_pdf
import pandas as pd
from tqdm import tqdm

def extract_text_pdf(file_path):
    """
    Extract the text from a pdf file given a system path as a string
    :param file_path: basestring Represents the file system path, complete path works better
    :return: basestring with all the text that was able to capture
    """

    class MyParser(object):
        def __init__(self, pdf):
            ## Snipped adapted from Yusuke Shinyamas
            # PDFMiner documentation
            # Create the document model from the file
            parser = PDFParser(open(pdf, "rb"))
            document = PDFDocument(parser)
            # Try to parse the document
            if not document.is_extractable:
                raise PDFTextExtractionNotAllowed
            # Create a PDF resource manager object
            # that stores shared resources.
            rsrcmgr = PDFResourceManager()
            # Create a buffer for the parsed text
            retstr = StringIO()
            # Spacing parameters for parsing
            laparams = LAParams()
            codec = "utf-8"

            # Create a PDF device object
            device = TextConverter(rsrcmgr, retstr, laparams=laparams)
            # Create a PDF interpreter object
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            # Process each page contained in the document.
            for page in PDFPage.create_pages(document):
                interpreter.process_page(page)

            self.records = []

            lines = retstr.getvalue().splitlines()
            for line in lines:
                self.handle_line(line)

        def handle_line(self, line):
            # Customize your line-by-line parser here
            self.records.append(line)

    a = MyParser(file_path)
    text = " ".join(a.records)
    return text


def extract_text_from_folder_of_pdfs(folder_path):
    """
    Scans the given folder for *.pdf files and extract all the text available
    :param folder_path: basestring represents the folder to scan for files
    :return: dict Dictionary with the file name being the key and the the text as value
    """
    all_files = glob.glob("%s/*.pdf" % (folder_path,))
    rtn_dict = dict.fromkeys(all_files)

    for pdf_file in tqdm(all_files):
        try:
            rtn_dict[pdf_file] = extract_text_pdf(pdf_file)
        except:
            pass

    return rtn_dict


def get_pdf_info(filename):
    """
    This function will return a dictionary containing metadata about a pdf
    
    Parameters:
    filename: pass the filename.
    """

    file = open(filename, "rb")
    pdf_reader = PdfFileReader(file)

    dic_info = dict(pdf_reader.getDocumentInfo())

    item = dic_info["/ModDate"]
    year = item[2:6]
    month = item[6:8]
    day = item[8:10]
    dic_info["year"] = year
    dic_info["month"] = month
    dic_info["day"] = day

    return dic_info


def pdf_folder_info(path, justdate=True):

    """
    This function will return additional information of a pdf (metadata) like date of creation.
    
    Parameters:
    path: pass the directory of the folder in which there are pdfs
    justdate: by default it returns only dates, if you want other metadata type justdate=False
    """

    diction = {}
    list_of_files = glob.glob(os.path.join(path, "*.pdf"))

    for file in list_of_files:
        infos = get_pdf_info(file)
        diction[file] = infos

    df_1 = pd.DataFrame(diction).T

    if justdate is True:
        df = df_1.drop(
            [
                "/Author",
                "/CreationDate",
                "/Creator",
                "/Keywords",
                "/ModDate",
                "/PTEX.Fullbanner",
                "/Producer",
                "/Subject",
                "/Title",
                "/Trapped",
            ],
            axis=1,
        ).reset_index()

        df.columns = ["name_of_pdf", "day", "month", "year"]
        return df

    if justdate is False:
        return df_1

    
from tabula import read_pdf
import glob
import pandas as pd
from tqdm import tqdm
import os


def extract_tables_greedy(folder_path=""):
    """
    This function will extract all tables from a folder containing pdf files.

    Parameters:
    folder_path: pass a directory in which there are pdf files.
    """
    files = glob.glob(folder_path + "*.pdf")

    for test_file in tqdm(files):

        name_folder = test_file.replace(".pdf","")

        if os.path.isdir(name_folder) is False:
                os.mkdir(name_folder)

        df = read_pdf(test_file, pages="all")

        for i in range(0, len(df)):

            df[i].to_csv(name_folder + "/table" + str(i+1) + ".csv")
            
    return df


