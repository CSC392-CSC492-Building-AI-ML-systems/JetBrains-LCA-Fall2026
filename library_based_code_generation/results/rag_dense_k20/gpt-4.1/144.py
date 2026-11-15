```python
import argparse
import sys
from pysilfont.ftml import FTML, FTMLBuilder
from pysilfont.fontinfo import getfontinfo, addfontinfo
from pysilfont.ufo import loadFont, openfont, Ufont
from pysilfont.glyphdata import readGlyphData

def main():
    parser = argparse.ArgumentParser(description="Generate FTML from UFO and glyph data CSV using pysilfont.")
    parser.add_argument("--ufo", required=True, help="Input UFO font directory")
    parser.add_argument("--out", required=True, help="Output FTML file")
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
    parser.add_argument("--totalwidth", type=int, default=0, help="Total width of all string column")
    parser.add_argument("--xsl", default=None, help="XSL stylesheet")
    args = parser.parse_args()

    # Load UFO font
    ufo = loadFont(args.ufo)
    fontinfo = getfontinfo(ufo)

    # Read glyph data CSV
    glyphdata = readGlyphData(args.glyphcsv)

    # Initialize FTML document
    ftml = FTML(
        testname=args.testname,
        fontsrc=args.fontsrc,
        xsl=args.xsl,
        log=args.log
    )
    builder = FTMLBuilder(ftml)

    # Add font info
    addfontinfo(ftml, fontinfo, fontcode=args.fontcode)

    # Add encoded characters
    builder.addEncodedChars(
        glyphdata,
        langs=args.langs,
        rtl=args.rtl,
        norendercheck=args.norendercheck,
        textscale=args.textscale,
        anchorre=args.anchorre,
        totalwidth=args.totalwidth
    )

    # Add unencoded specials and ligatures
    builder.addUnencodedSpecials(glyphdata)
    builder.addLigatures(glyphdata)

    # Add Lam-Alef data if present
    if "lamalef" in glyphdata:
        builder.addLamAlef(glyphdata["lamalef"])

    # Add diacritic attachment data if present
    if "diacritics" in glyphdata:
        builder.addDiacriticAttachments(glyphdata["diacritics"])

    # Write FTML output
    with open(args.out, "w", encoding="utf-8") as f:
        ftml.write(f)

if __name__ == "__main__":
    main()
```