"""
This file will extract raw textual information from pdf files located in the working directory and save it in .txt files.
"""

from tika import parser
import os
import zipfile
import glob
from datetime import datetime


def second_wrapper():

    def extract_text_pdf(file_name):
        """
        Extract a text from a given pdf file (full path required)
        """

        rawText = parser.from_file(file_name)
        text_l = rawText['content'].splitlines()

        clean_txt = ' '.join(word for word in text_l)

        return clean_txt


    # cleans text
    def txt_clean(word_list, min_len):
        clean_words = []
        for line in word_list:
            parts = line.strip().split()
            for word in parts:
                word_l = word.lower()
                #print (word_l, '\n', '\n')
                if word_l.isalpha():
                    if len(word_l) > min_len:
                        clean_words.append(word_l)
        return clean_words


    def extract_text_from_folder_of_pdfs(folder_path, txt=True):
        """
        Scans the given folder for *.pdf files and extract all the text available
        :param folder_path: basestring represents the folder to scan for files
        :return: dict Dictionary with the file name being the key and the the text as value
        """
        all_files = glob.glob("%s/*.pdf" % (folder_path,))
        rtn_dict = dict.fromkeys(all_files)
        #print (rtn_dict)

        folder_path_txt = folder_path + '/txt'
        if not os.path.exists (folder_path + '/txt'):
            os.makedirs (folder_path_txt)
        
        for pdf_file in all_files:
            try:
                rtn_dict[pdf_file] = extract_text_pdf(pdf_file)
                #print('\n', pdf_file, ': done')
            except Exception as e:
                #print('\n', pdf_file, ': pdf is not readable')
                pass



        if txt==True:
            txt_file_list = []
            
            for i in rtn_dict.keys():
                try:
                    txt_i = (i.replace(folder_path, folder_path_txt)).replace(".pdf",".txt")
                    txt_file_list.append(txt_i)
                    text_file = open(txt_i, "w")
                    text_file.write(rtn_dict[i])
                    text_file.close()
                except:
                    pass

        # asking the user if they want a zip file to be generated
        zip_y_n = input ('\ndo you want a zip file to be generated from the txt files? (y/n) ')

        if zip_y_n == 'y':
            # generating "output.zip" file with txt files in the specified folder path
            get_zip(folder_path_txt)
            print ('\n-zip file generated\n')


        return rtn_dict

    def get_zip(folder_path):
        """
        This function will return a zipfile containing all .txt outputs generated in the specified folder.
        """

        file_list = glob.glob(folder_path + '/*.txt')

        if len(file_list):
            with zipfile.ZipFile(folder_path + '/all_texts.zip', 'w') as zip:
                for file_name in file_list:
                    zip.write(file_name)


    # Main --------------------------------------

    print ('\n-Starting the pdf to txt conversion process-\n')
    start_time = datetime.now()
    print('--- starting time: {}'.format(start_time))


    # asking the user for the the folder path name
    #print ('\nplease enter the path of the folder where your pdf files are')

    folder_path = 'doc_folder'

    # extracting text from all .pdf files in the specified folder path
    extract_text_from_folder_of_pdfs(folder_path)


    print ('\n--- this is the end of the process ---\n')

    print('--- total duration: {}'.format(datetime.now() - start_time))