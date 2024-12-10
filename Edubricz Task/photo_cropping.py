import cv2
from PIL import Image, ImageOps
import os

input_image_path = r"C:\Users\SANGEETHA T SHENOY\Desktop\Edubricz Task\image.jpg"
output_directory = r"C:\Users\SANGEETHA T SHENOY\Desktop\Edubricz Task"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

image = cv2.imread(input_image_path)


if image is None:
    print("Error: Could not load the image. Check the file path.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

print(f"Found {len(faces)} face(s) in this image.")

passport_size = (413, 531)

for i, (x, y, w, h) in enumerate(faces):
    try:
    
        padding = int(0.4 * h)  
        new_y = max(0, y - padding) 
        new_h = h + padding + int(0.2 * h)  
        new_x = max(0, x - int(0.2 * w)) 
        new_w = w + int(0.4 * w)  
       
        cropped_face = image[new_y:new_y + new_h, new_x:new_x + new_w]
        
        passport_template = Image.new("RGBA", passport_size, (255, 255, 255, 0))
        
        pil_image = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGBA))
        
        pil_image = ImageOps.contain(pil_image, passport_size)

        offset_x = (passport_size[0] - pil_image.width) // 2
        offset_y = (passport_size[1] - pil_image.height) // 2
        passport_template.paste(pil_image, (offset_x, offset_y), pil_image)

        output_path = f"{output_directory}\\passport_photo_{i + 1}.png"
        passport_template.save(output_path)
        print(f"Saved passport photo {i + 1} at {output_path}")

    except Exception as e:
        print(f"Error processing face {i + 1}: {e}")
