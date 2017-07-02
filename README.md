# Handwriting Recognition Data Generator
`Handwriting Recognition Data Generator` is a utility that generates images that attempt to look like handwritten documents. The user can specify images to use as background and a source of glyphs images that will be stamped onto the page.

## Getting started
```bash
pip install -r requirements.txt
python main.py -h
```

* Glyphs are expected to be located in folders named after their decimal representation (e.g., A => 65, a => 97)
* Glyphs for the same font are expected to have the same name in each folder (e.g., 65/00001.png, 97/00001.png)

## Limitations
* Only a subset of Latin/English glyphs (! (ASCII 33) to ~ (ASCII 126))
* Only a single region (where glyphs are stamped)
* Glyph images are expected to be jpg or png

## Configuration
```yml
backgrounds:
  -
    file: /path/to/file
    regions:
      [[100, 100], [500, 500]]
    line_height: 50
    dilate: 3
    blur: 0
```

* `file`: location of the background image on disk
* `regions`: array of arrays defining the regions where glyphs may be stamped, in the form (top left x, top left y, bottom right x, bottom right y)
* `line_height`: the height (in pixels) of the line in the background image
* `dilate`: amount of morphological dilate to apply to glyphs
* `blur`: amount of blur to apply to glyphs

## License
The code is licensed under the [MIT license](http://choosealicense.com/licenses/mit/). See [LICENSE](LICENSE).
