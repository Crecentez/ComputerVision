from PIL import Image, ImageColor
import os
# from EdgeDetection import EdgeDetection
from ImageProcessor import ImageProcessor
import time
import tkinter as tk
from tkinter import filedialog
import math
import numpy as np



root = tk.Tk()
root.withdraw()

ImagePath = filedialog.askopenfilename(title="Select an Image", filetypes=[("All Files", "*.*")])

# folder = "Blurred"
# folder = "Edges"
folder = "Edges_FT"

GreyScale_folder = "GreyScale"
Blurred_folder = "Blurred"
Edge_Folder = "Edges"
FT_Edge = "Edges_FT"



name = os.path.splitext(os.path.basename(ImagePath))[0]

def GetPath(folderName: str) -> str:
    return os.path.dirname(os.path.abspath(__file__)) + "\\TestImages\\" + folderName + "\\" + name + ".png"

if ImagePath:
    start = math.floor(time.time() * 100) / 100
    print(f"Process running...")
    
    
    image = Image.open(ImagePath).convert("L")
    print("Getting array from image...")
    imageArr: np.ndarray = ImageProcessor.GetArr(image)
    ImageProcessor.SaveImageArr(imageArr, GetPath(GreyScale_folder))
    print("Applying Gaussian Filter...")
    imageArr: np.ndarray = ImageProcessor.ApplyGaussianFilter(imageArr, 2, 2)
    ImageProcessor.SaveImageArr(imageArr, GetPath(Blurred_folder))
    print("Getting Edges...")
    imageArr, edge_test = ImageProcessor.GetO(imageArr)
    ImageProcessor.SaveImageArr(edge_test, GetPath(Edge_Folder))
    ImageProcessor.SaveImageArr(imageArr, GetPath(FT_Edge))
    
    print(f"Process took {(math.floor(time.time() * 100) / 100) - start} seconds to run.")
    input("Press enter to stop...")
else:
    print("Image not found")

# ImagePath = os.path.dirname(os.path.abspath(__file__)) + "\\TestImages\\Flower.jpg"
# image = Image.open(ImagePath)
# # image.show()

# print(EdgeDetection.getGreyScale(image, 1, 1))

# start = time.time()
# edgeDetection = EdgeDetection.getG(image)

# scaled_array = [[int(value * 255) for value in row] for row in edgeDetection]

# print(f"Took {time.time() - start} seconds to complete")
# height = len(scaled_array)
# width = len(scaled_array[0])
# Sobel_Image = Image.new("L", (width, height))
# for y in range(height):
#     for x in range(width):
#         Sobel_Image.putpixel((x, y), scaled_array[y][x])
# Sobel_Image.save(f"{os.path.dirname(os.path.abspath(__file__))}\\TestImages\\Sobel_avg_Flower.png")
# Sobel_Image.show()
# input("Press enter to stop...")
