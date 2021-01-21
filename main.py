from utils import *
from files import *
import glob
import shutil

print("Converting pdf to images...")
pdf_to_image("sample.pdf")
paths = sorted(glob.glob("images/*"))
os.mkdir("output")

print("Enhancing images...\n")
for p in paths:
  img_name = p.split("/")[-1]
  image = cv2.imread(p)
  try:
    blurred_threshold = transformation(image)
    cleaned_image = final_image(blurred_threshold,120)
    cv2.imwrite(p.replace("images","output"), cleaned_image
    )
    print("Enhancing of image ",img_name," is finished!")
  except:
    print("!!!        Error in enhancing image ",img_name,"     !!!")


print("Compiling enhanced images to pdf...")
image_to_pdf("output.pdf")

print("Removing temprory folders...")

dirs = ['images','output']

for dir in dirs:
  try:
      shutil.rmtree(dir)
  except OSError as e:
      print("Error: %s : %s" % (dir, e.strerror))


print("Conversion finished successfully!")