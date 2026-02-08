# reap

`reap` is a low-level PDF parser written in Rust, with Python bindings for fast geometry-aware text extraction and spatial queries.
It is designed for fast spatial lookups, but also supports full text extraction and regex queries.

## Installing
```sh
uv add reap-pdf
```

## Usage
```py
from reap import Rectangle, TextBlockIndex

index = TextBlockIndex.from_path("my_w2.pdf")
blocks = index.search_regex("Wage and Tax Statement")
print(blocks)
#> [TextBlock("Wage and Tax Statement")]

# Search to the right of the label.
search_rect = Rectangle(
    top=blocks[0].rect.top,
    left=blocks[0].rect.right,
    bottom=blocks[0].rect.bottom,
    right=blocks[0].rect.right + 100,
)
year = index.search(search_rect, overlap=0.3)
print(year[0].text)
#> 2026
```

## Details
* `TextBlockIndex` builds an index of word level `TextBlock`s, across all pages.
* `TextBlockIndex.search_regex` works on the entire text corpus, in reading order, but returns merged `TextBlock`s that match the query into one `TextBlock`.
* `TextBlockIndex.text` returns text in reading order.
* All pages are merged, and considered as one big page, with each page's coordinates stacked below each other. Coordinate units are points.

## Advanced use cases
* `TextBlockIndex.scoped` creates a `TextBlockIndex` scoped to a specific region, with support for merging text blocks within the region into multi-word `TextBlock`s.
* `TextBlockIndex(..., include_chars=True)` enables `TextBlockIndex.chars`, returning all `TextChar`s for each page.

## Scope
* OCR is not in-scope. Reap focuses entirely on extracting visible text present in the PDF.
* No promises for all PDF spec details to be supported, support is added as the need arises.
* Best effort to extract all *visible* text in PDFs.

## Comparisons
### Extraction differences
`PyMuPDF` and `pdfminer` will both extract hidden text, eg. white text on a white background, which is often not desirable.
`reap` performs visibility checks during extraction, to ensure that invisible text does not get included in the final output.

### Speed
Text extraction performance measured for comparability, as neither alternative offers proper spatial query support.
| Library  | Text Extraction              |
|----------|------------------------------|
| reap     | 0.850ms median               |
| PyMuPDF  | 5ms median                   |
| pdfminer | 220ms median                 |

Speed comparisons vary greatly depending on the PDF, with smaller PDFs `reap` is roughly
10x faster than `PyMuPDF`, and 500x faster than `pdfminer`, with larger PDFs the advantage narrows to roughly 5x and 250x respectively.
Tests were conducted on a private test suite of PDFs with varying complexity