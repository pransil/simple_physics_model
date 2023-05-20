# Write data to file data.txt
# Open file data.txt and write data to it
# Each line contains 3 points, (x1, y1), (x2, y2), (x3, y3)
# x1, y1 are random points in range [-1.0 to 1.0]
# vx and vy are random velocity in range [-1.0 to 1.0]
# x2 = x1 + vx
# y2 = y1 + vy - 0.5 * g
# x3 = x2 + vx
# y3 = y1 + vy - 0.5 * g * 4
# Write N lines to file data.txt in the format:
# x1 y1 x2 y2 x3 y3 vx vy g
# close file data.txt

import random

g = 0.4 # gravity - small for testing

def make_data(lines):
    f = open("data.txt", "w")
    # First line should say "format: x1 y1 x2 y2 x3 y3, lines lines of data"
    f.write("# format: x1 y1 x2 y2 x3 y3 vx vy g " + str(lines) + " lines of data\n")
    for i in range(lines):
        x1 = random.uniform(-0.5, 0.5)
        y1 = random.uniform(-0.5, 0.5)
        vx = random.uniform(-0.5, 0.5)
        vy = random.uniform(-0.5, 0.5)
        
        x2 = x1 + vx
        y2 = y1 + vy - 0.5 * g
        x3 = x2 + vx
        y3 = y1 + vy - 0.5 * g * 4
        f.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + \
                " " + str(x3) + " " + str(y3) + " " +  str(vx) + " " + str(vy) + " " + str(g) + "\n")
    f.close()

make_data(1000)