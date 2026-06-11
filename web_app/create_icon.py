"""Generates a realistic stock chart favicon saved as web_app/icon.png."""
from PIL import Image, ImageDraw

SIZE = 64
img = Image.new("RGBA", (SIZE, SIZE), (15, 17, 23, 255))  # dark background
draw = ImageDraw.Draw(img)

# Grid lines
for y in [16, 32, 48]:
    draw.line([(4, y), (60, y)], fill=(40, 44, 52, 180), width=1)

# Candlestick data: (x, low, high, open, close)
candles = [
    (10, 38, 52, 45, 40),
    (18, 32, 48, 42, 46),
    (26, 28, 44, 38, 30),
    (34, 24, 40, 32, 38),
    (42, 18, 36, 28, 34),
    (50, 14, 30, 22, 16),
]

for x, low, high, open_, close in candles:
    color = (46, 204, 113, 255) if close >= open_ else (231, 76, 60, 255)
    # Wick
    draw.line([(x, low), (x, high)], fill=color, width=1)
    # Body
    top = min(open_, close)
    bot = max(open_, close)
    draw.rectangle([(x - 3, top), (x + 3, bot)], fill=color)

img.save("web_app/icon.png")
print("Icon saved to web_app/icon.png")
