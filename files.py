from pdf2image import convert_from_path
import img2pdf
from PIL import Image
import os
import glob

def pdf_to_image(pdf_file_name: str)-> str:
  """ Convert a PDF to an image with a 2 x 4 panel """
  images = convert_from_path(pdf_file_name)  
  num_page = len(images)
  pg = 0
  os.mkdir("images")
  for img in images: 
    pg += 1

    if pg <10:
      page = "0"+str(pg)
    else:
      page = str(pg)

    img.save("images/"+page+'.jpg', 'JPEG')

def image_to_pdf(pdf_file_name: str)-> str:
  """ Compile images to pdf """
  with open(pdf_file_name,"wb") as f:
	  f.write(img2pdf.convert(sorted(glob.glob("output/*"))))
