```python
import argparse
import sys
import re
from pysilfont.ufo import Font
from pysilfont.ftml import FTMLBuilder
from pysilfont.util import readGlyphData

def main():
    parser = argparse.ArgumentParser(description="Generate FTML from UFO and glyph data CSV using pysilfont.")
    parser.add_argument("--ufo", required=True, help="Input UFO font directory")
    parser.add_argument("--out", required=True, help="Output FTML file")
    parser.add_argument("--csv", required=True, help="Glyph info CSV file")
    parser.add_argument("--fontcode", default="", help="Font code")
    parser.add_argument("--log", default=None, help="Log file name")
    parser.add_argument("--langs", nargs="*", default=[], help="List of BCP47 language tags")
    parser.add_argument("--rtl", action="store_true", help="Enable right-to-left feature")
    parser.add_argument("--norendercheck", action="store_true", help="Disable rendering check")
    parser.add_argument("--testname", default="Test", help="Test name")
    parser.add_argument("--fontsrc", default="", help="Font source")
    parser.add_argument("--textscale", type=float, default=1.0, help="Text scaling factor")
    parser.add_argument("--anchorre", default=None, help="Anchor points regular expression")
    parser.add_argument("--totalwidth", type=int, default=0, help="Total width of all string column")
    parser.add_argument("--xsl", default=None, help="XSL stylesheet")
    args = parser.parse_args()

    # Read glyph data from CSV
    glyph_data = readGlyphData(args.csv)

    # Load UFO font
    font = Font(args.ufo)

    # Initialize FTMLBuilder
    ftml = FTMLBuilder(
        testname=args.testname,
        fontsrc=args.fontsrc,
        fontcode=args.fontcode,
        langs=args.langs,
        rtl=args.rtl,
        norendercheck=args.norendercheck,
        textscale=args.textscale,
        anchorre=args.anchorre,
        totalwidth=args.totalwidth,
        xsl=args.xsl
    )

    # Add encoded characters
    for glyph in glyph_data.get("encoded", []):
        ftml.addToTest(
            glyph=glyph,
            font=font,
            testtype="encoded",
            langs=args.langs
        )

    # Add unencoded specials and ligatures
    for glyph in glyph_data.get("specials", []):
        ftml.addToTest(
            glyph=glyph,
            font=font,
            testtype="special",
            langs=args.langs
        )
    for glyph in glyph_data.get("ligatures", []):
        ftml.addToTest(
            glyph=glyph,
            font=font,
            testtype="ligature",
            langs=args.langs
        )

    # Add Lam-Alef data if present
    if "lamalef" in glyph_data:
        for glyph in glyph_data["lamalef"]:
            ftml.addToTest(
                glyph=glyph,
                font=font,
                testtype="lamalef",
                langs=args.langs
            )

    # Add diacritic attachment data if present
    if "diacritics" in glyph_data:
        anchorre = re.compile(args.anchorre) if args.anchorre else None
        for glyph in glyph_data["diacritics"]:
            if not anchorre or anchorre.match(glyph):
                ftml.addToTest(
                    glyph=glyph,
                    font=font,
                    testtype="diacritic",
                    langs=args.langs
                )

    # Write FTML to output file
    with open(args.out, "w", encoding="utf-8") as f:
        ftml.writeFile(f)

if __name__ == "__main__":
    main()
```