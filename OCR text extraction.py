# -*- coding: utf-8 -*-
"""
Text extraction: PDF to CSV data

@author: Stefano Gioia
"""
#pip install pdf2image
#pip install pytesseract
#poppler in path (environment variable)

import pdf2image
import pytesseract
from pytesseract import Output, TesseractError
#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe' #to run pytesseract
import pandas as pd
import csv 
     

datapath = "C:/User/Downloads/pdfspath/"
outcsvpath = "C:/User/Downloads/output_csv/" #output csv files


import glob, os
# i=0
os.chdir(datapath)
for pdfname in glob.glob("*.pdf"):
    pdf_path = datapath + pdfname
    # print(file)
    # i+=1
    # print(i) #check
    
    images = pdf2image.convert_from_path(pdf_path)
    
    
    pil_im = images[0] # first page only
    
    #OCR info including text and location on the image
    ocr_dict = pytesseract.image_to_data(pil_im, lang='eng', output_type=Output.DICT) #e.g. lang='deu': German
    
    main_df = pd.DataFrame(ocr_dict) #convert to dataframe
    
    #record indexes
    main_df['prev_idx'] = range(0, len(main_df))
    
    #Only nonempty text cells
    ndf= main_df.loc[main_df['word_num']!=0]

    
    #new indexes
    ndf=ndf.reset_index(drop=True)
    
    csv_file = outcsvpath + pdfname[:-4] + ".csv"  
    
    ndf.to_csv(csv_file)
    
 