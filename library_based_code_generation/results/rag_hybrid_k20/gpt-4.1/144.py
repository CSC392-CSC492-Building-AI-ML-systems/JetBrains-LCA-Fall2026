```python
import argparse
import sys
from pysilfont.ftml import FTML, FTMLBuilder, readGlyphData
from pysilfont.fontinfo import FontInfo
from pysilfont.ufo import loadFont
from pysilfont.util import setFileForOutput

def main():
    parser = argparse.ArgumentParser(description="Generate FTML from UFO and glyph data CSV.")
    parser.add_argument("--ufo", required=True, help="Input UFO font directory")
    parser.add_argument("--output", required=True, help="Output FTML file")
    parser.add_argument("--glyphcsv", required=True, help="Glyph info CSV file")
    parser.add_argument("--fontcode", default="", help="Font code")
    parser.add_argument("--log", default=None, help="Log file name")
    parser.add_argument("--langs", nargs="*", default=[], help="List of BCP47 language tags")
    parser.add_argument("--rtl", action="store_true", help="Enable right-to-left feature")
    parser.add_argument("--norendercheck", action="store_true", help="Disable rendering check")
    parser.add_argument("--testname", default="Test", help="Test name")
    parser.add_argument("--fontsource", default="", help="Font source")
    parser.add_argument("--textscale", type=float, default=1.0, help="Text scaling factor")
    parser.add_argument("--anchorre", default=None, help="Anchor points regular expression")
    parser.add_argument("--totalwidth", type=int, default=0, help="Total width of all string column")
    parser.add_argument("--xsl", default=None, help="XSL stylesheet")
    args = parser.parse_args()

    # Load font and glyph data
    font = loadFont(args.ufo)
    fontinfo = FontInfo(font)
    glyphdata = readGlyphData(args.glyphcsv)

    # Initialize FTML document
    ftml = FTML(
        testname=args.testname,
        fontsrc=args.fontsource,
        fontinfo=fontinfo,
        fontcode=args.fontcode,
        langs=args.langs,
        rtl=args.rtl,
        norendercheck=args.norendercheck,
        textscale=args.textscale,
        anchorre=args.anchorre,
        totalwidth=args.totalwidth,
        xsl=args.xsl
    )

    builder = FTMLBuilder(ftml, font, glyphdata)

    # Add encoded characters
    builder.addEncodedChars()

    # Add unencoded specials and ligatures
    builder.addUnencodedSpecials()
    builder.addLigatures()

    # Add Lam-Alef data
    builder.addLamAlefData()

    # Add diacritic attachment data
    builder.addDiacriticAttachmentData()

    # Write FTML to output file
    with setFileForOutput(args.output, "w", encoding="utf-8") as f:
        ftml.write_to_file(f)

if __name__ == "__main__":
    main()
```