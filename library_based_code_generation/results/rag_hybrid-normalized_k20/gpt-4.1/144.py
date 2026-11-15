```python
import argparse
import sys
import re
from pysilfont.ftml import FTML, FTMLBuilder
from pysilfont.fontinfo import FontInfo
from pysilfont.glyphdata import readGlyphData
from pysilfont.ufo import loadFont

def main():
    parser = argparse.ArgumentParser(description="Generate FTML from UFO and glyph data CSV using pysilfont.")
    parser.add_argument("--ufo", required=True, help="Input UFO font directory")
    parser.add_argument("--output", required=True, help="Output FTML file")
    parser.add_argument("--glyphcsv", required=True, help="Glyph info CSV file")
    parser.add_argument("--fontcode", required=True, help="Font code")
    parser.add_argument("--log", default=None, help="Log file name")
    parser.add_argument("--langs", nargs='*', default=[], help="List of BCP47 language tags")
    parser.add_argument("--rtl", action="store_true", help="Enable right-to-left feature")
    parser.add_argument("--norendercheck", action="store_true", help="Disable rendering check")
    parser.add_argument("--testname", default="Test", help="Test name")
    parser.add_argument("--fontsrc", default=None, help="Font source")
    parser.add_argument("--textscale", type=float, default=1.0, help="Text scaling factor")
    parser.add_argument("--anchorre", default=None, help="Anchor points regular expression")
    parser.add_argument("--totalwidth", type=int, default=None, help="Total width of all string column")
    parser.add_argument("--xsl", default=None, help="XSL stylesheet")
    args = parser.parse_args()

    # Load UFO font
    font = loadFont(args.ufo)

    # Read glyph data CSV
    glyph_data = readGlyphData(args.glyphcsv)

    # Initialize FTML document
    ftml = FTML(
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

    # Add font info
    fontinfo = FontInfo(font)
    ftml.addFont(fontinfo)

    # Initialize FTMLBuilder
    builder = FTMLBuilder(ftml, font, glyph_data)

    # Add encoded characters
    builder.addEncodedChars()

    # Add unencoded specials and ligatures
    builder.addUnencodedSpecials()
    builder.addLigatures()

    # Add Lam-Alef data if present
    if hasattr(builder, "addLamAlefData"):
        builder.addLamAlefData()

    # Add diacritic attachment data if present
    if hasattr(builder, "addDiacriticAttachmentData"):
        builder.addDiacriticAttachmentData()

    # Write FTML to output file
    ftml.writeFile(args.output)

if __name__ == "__main__":
    main()
```