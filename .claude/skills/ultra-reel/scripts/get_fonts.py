"""Download full TTFs (all scripts incl. Cyrillic / Kazakh) from the Google Fonts CSS API.

Usage: get_fonts.py OUT_DIR "Family:axis@values" ...
Files are named Family-<weight>[i].ttf, e.g. Oswald-700.ttf, PlayfairDisplay-700i.ttf.
"""
import os
import re
import sys
import urllib.request

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for fam in sys.argv[2:]:
    name, _, spec = fam.partition(":")
    url = f"https://fonts.googleapis.com/css2?family={name.replace(' ', '+')}" + (f":{spec}" if spec else "")
    # a non-browser UA makes the API answer with plain TrueType files covering every subset
    css = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8.0"}), timeout=30).read().decode()
    got = 0
    for block in re.findall(r"@font-face\s*{(.*?)}", css, re.S):
        w = re.search(r"font-weight:\s*(\d+)", block)
        st = re.search(r"font-style:\s*(\w+)", block)
        u = re.search(r"url\((.*?)\)", block)
        if not u:
            continue
        fn = f"{name.replace(' ', '')}-{w.group(1) if w else '400'}{'i' if st and st.group(1) == 'italic' else ''}.ttf"
        path = os.path.join(out, fn)
        if not os.path.exists(path):
            open(path, "wb").write(urllib.request.urlopen(u.group(1), timeout=60).read())
            got += 1
    print(f"{name}: {got} new file(s)")
