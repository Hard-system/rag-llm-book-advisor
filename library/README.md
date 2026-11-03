# Put your books here

Drop any supported file into this folder: PDF, EPUB, DOCX, TXT, MD, HTML or IPYNB. Then run:

```powershell
rag sync
```

* **Sub-folders become categories**, which you can use as search filters:

  ```
  library/
    psychology/   Daniel Kahneman - Thinking, Fast and Slow.pdf
    finance/      Робърт Кийосаки - Богат татко, беден татко.epub
    health/       Matthew Walker - Why We Sleep.epub
  ```

* **File name `Author - Title.ext`** is used when the file has no embedded metadata.
* **Optional metadata override.** Put a file named `<book file name>.yaml` next to the book:

  ```yaml
  title: Thinking, Fast and Slow
  author: Daniel Kahneman
  language: en
  tags: [psychology, decision-making]
  ```

* Files and folders starting with `.`, `_` or `~$` are ignored. Use a folder like `_inbox` for books you don't want indexed yet.
* Scanned PDFs (images only) need OCR first, for example `ocrmypdf input.pdf output.pdf`.

The books in this folder are **never committed to git** (see `.gitignore`).
