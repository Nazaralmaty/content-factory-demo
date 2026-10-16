"""Contact sheet of PNG stills. Usage: contact.py STILLS_DIR out.png [cols=6] [start=0] [count=all]"""
import sys, os
from PIL import Image, ImageDraw, ImageFont
files = sorted(os.listdir(sys.argv[1])); files = [f for f in files if f.endswith('.png')]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 6
tw, th = 300, 533
start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
count = int(sys.argv[5]) if len(sys.argv) > 5 else len(files)
files = files[start:start+count]
rows = (len(files) + cols - 1) // cols
sheet = Image.new('RGB', (cols * tw, rows * (th + 28)), (40, 40, 40))
d = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    im = Image.open(os.path.join(sys.argv[1], f)).convert('RGB').resize((tw, th), Image.LANCZOS)
    x, y = (i % cols) * tw, (i // cols) * (th + 28)
    sheet.paste(im, (x, y + 28))
    d.text((x + 6, y + 6), f, fill=(255, 255, 255))
sheet.save(sys.argv[2])
