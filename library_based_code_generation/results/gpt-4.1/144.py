```python
#!/usr/bin/env python3

import argparse
import csv
import re
import sys
from pathlib import Path
from pysilfont.ftml import FTML
from defcon import Font

def parse_args():
    parser = argparse.ArgumentParser(description="Generate FTML from UFO and glyph data CSV using pysilfont.")
    parser.add_argument("--ufo", required=True, help="Input UFO font directory")
    parser.add_argument("--out", required=True, help="Output FTML file")
    parser.add_argument("--csv", required=True, help="Glyph info CSV file")
    parser.add_argument("--fontcode", default="", help="Font code")
    parser.add_argument("--log", default=None, help="Log file name")
    parser.add_argument("--langs", default="", help="Comma-separated list of BCP47 language tags")
    parser.add_argument("--rtl", action="store_true", help="Enable right-to-left feature")
    parser.add_argument("--norendercheck", action="store_true", help="Disable rendering check")
    parser.add_argument("--testname", default="Test", help="Test name")
    parser.add_argument("--fontsrc", default="", help="Font source")
    parser.add_argument("--textscale", type=float, default=1.0, help="Text scaling factor")
    parser.add_argument("--anchorre", default=".*", help="Anchor points regular expression")
    parser.add_argument("--totalwidth", type=int, default=1000, help="Total width of all string column")
    parser.add_argument("--xsl", default=None, help="XSL stylesheet")
    return parser.parse_args()

def read_csv(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]

def main():
    args = parse_args()

    # Logging setup
    if args.log:
        sys.stderr = open(args.log, "w")

    # Load UFO font
    font = Font(args.ufo)

    # Read glyph data CSV
    glyph_data = read_csv(args.csv)

    # Parse language tags
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]

    # Compile anchor regex
    anchor_re = re.compile(args.anchorre)

    # Initialize FTML document
    ftml = FTML(
        testname=args.testname,
        fontsrc=args.fontsrc,
        fontcode=args.fontcode,
        langs=langs,
        rtl=args.rtl,
        norendercheck=args.norendercheck,
        textscale=args.textscale,
        anchorre=anchor_re,
        totalwidth=args.totalwidth,
        xsl=args.xsl
    )

    # Add font to FTML
    ftml.addFont(args.ufo, fontcode=args.fontcode)

    # Process glyph data
    for row in glyph_data:
        glyphname = row.get("glyphname") or row.get("name")
        if not glyphname:
            continue

        # Encoded characters
        if "unicode" in row and row["unicode"]:
            try:
                codepoint = int(row["unicode"], 16)
                char = chr(codepoint)
                ftml.addEncoded(char, glyphname=glyphname)
            except Exception:
                pass

        # Unencoded specials and ligatures
        if row.get("special") == "1":
            ftml.addSpecial(glyphname)

        if row.get("ligature") == "1":
            lig_components = row.get("components", "").split()
            if lig_components:
                ftml.addLigature(glyphname, lig_components)

        # Lam-Alef data
        if row.get("lamalef") == "1":
            lam = row.get("lam")
            alef = row.get("alef")
            if lam and alef:
                ftml.addLamAlef(glyphname, lam, alef)

        # Diacritic attachment data
        if row.get("diacritic") == "1":
            base = row.get("base")
            anchor = row.get("anchor")
            if base and anchor and anchor_re.match(anchor):
                ftml.addDiacritic(glyphname, base, anchor)

    # Write FTML output
    with open(args.out, "w", encoding="utf-8") as outf:
        outf.write(ftml.toXML(pretty=True))

if __name__ == "__main__":
    main()
```