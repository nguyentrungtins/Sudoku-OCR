import cv2
from imutils import contours
import numpy as np

# Load image, grayscale, and adaptive threshold
image = cv2.imread('./board.png')
cv2.imshow("raw", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,57,5)
# Filter out all numbers and noise to isolate only boxes
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 1000:
        cv2.drawContours(thresh, [c], -1, (0,0,0), -1)
# Fix horizontal and vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
threshGrid = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)
# Sort by top to bottom and each row by left to right
invert = 255 - threshGrid
cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

sudoku_rows = []
row = []
for (i, c) in enumerate(cnts, 1):
    area = cv2.contourArea(c)
    if area < 50000:
        row.append(c)
        if i % 9 == 0:  
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            sudoku_rows.append(cnts)
            row = []

# Iterate through each box
for row in sudoku_rows:
    for c in row:
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, (255,255,255), -1)
        result = cv2.bitwise_and(image, mask)
        result[mask==0] = 255
        cv2.imshow('Finding Number', result)
        cv2.waitKey(150)

def getNumber(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2);
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2);

    cv2.imshow("thresh", thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bbox = [cv2.boundingRect(c) for c in contours]

    # filter rectangles by width and height
    height = img.shape[0]
    width = img.shape[1]
    scale = 0.95
    max_w = scale * width 
    min_w = 0.1 * width
    max_h = scale * height 
    min_h = 0.5 * height
    isNumber = False
    for x, y, w, h in bbox:
        if (4 < w < max_w) and (min_h < h < max_h):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            isNumber = True
            break
    return img, isNumber

number = 0
rec_list = []
board = []
for row in sudoku_rows:
    board_row = []
    for c in row:
        x,y,w,h = cv2.boundingRect(c)
        crop_img = image[y:y+h, x:x+w]
        # rec_list.append(crop_img)
        img_result, isNumber = getNumber(crop_img)
        if (isNumber):
            board_row.append(1)
        else: 
            board_row.append(0)
        # cv2.imshow('rec', img_result) 
        # cv2.waitKey(0)
    board.append(board_row)
# cv2.imshow('thresh', thresh)
# cv2.imshow('invert', invert)
def niceSudo(board):
    side    = len(board)
    base    = int(side**0.5)
    def expandLine(line):
        return line[0]+line[5:9].join([line[1:5]*(base-1)]*base)+line[9:]
    line0  = "  "+expandLine("╔═══╤═══╦═══╗")
    line1  = "# "+expandLine("║ . │ . ║ . ║ #")
    line2  = "  "+expandLine("╟───┼───╫───╢")
    line3  = "  "+expandLine("╠═══╪═══╬═══╣")
    line4  = "  "+expandLine("╚═══╧═══╩═══╝")

    symbol = " 123456789" if base <= 3 else " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    nums   = [ [""]+[f"({symbol[-n]})" if n<0 else f" {symbol[n]} "  for n in row]
               for row in board ]
    coord  = "   "+"".join(f" {s}  " for s in symbol[1:side+1])
    lines  = []
    lines.append(coord)
    lines.append(line0)
    for r in range(1,side+1):
        line1n = line1.replace("#",str(symbol[r]))
        lines.append( "".join(n+s for n,s in zip(nums[r-1],line1n.split(" . "))) )
        lines.append([line2,line3,line4][(r%side==0)+(r%base==0)])
    lines.append(coord)
    print(*lines,sep="\n")
    with open('results.txt', 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines))
niceSudo(board)
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

