# DocFinder RAG

A small, fully-offline desktop app that indexes **your own documents** and lets
you ask questions over them in plain language. It retrieves the most relevant
passages (Retrieval-Augmented Generation) and shows you an answer together with
the exact source text it came from — handy for "where did I note down the login
for X?", "what's our password policy?", and similar searches across your own
notes, PDFs, and docs.

> It searches documents **you** provide. It does not guess, crack, or recover
> anyone's passwords.

## Features
- Index a folder or individual files — **PDF, DOCX, TXT, MD**.
- Natural-language search using TF-IDF retrieval (runs entirely on your machine,
  no internet, no model downloads).
- Answers are **extractive by default**: the exact sentences from your documents,
  with the source files and relevance scores shown below.
- Optional LLM synthesis: add an OpenAI or Anthropic API key in **Settings** to
  get a written answer composed from the retrieved passages. Off by default;
  nothing leaves your machine unless you configure a key.
- Your index is saved to `~/.docfinder_rag/` and reloaded on next launch.

## Run from source
```bash
python -m pip install -r requirements.txt
python app.py
```

## Build the .exe
Double-click **`build_exe.bat`**, or run:
```bash
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --onefile --windowed --name DocFinderRAG ^
  --collect-submodules sklearn --collect-submodules scipy app.py
```
The result is `dist\DocFinderRAG.exe` — a single file you can double-click or
copy to another Windows PC (no Python needed there).

## Using it
1. Click **Folder…** or **Files…** and pick your documents.
2. Click **Build Index** (progress shows in the status line).
3. Type a question and press **Enter** or **Ask**.
4. Read the answer up top; the passages it came from are listed below with their
   file paths.

## Files
| File | Purpose |
|------|---------|
| `app.py` | Tkinter GUI |
| `rag_core.py` | Document loading, chunking, TF-IDF index, retrieval, answers |
| `build_exe.bat` | One-click PyInstaller build |
| `requirements.txt` | Python dependencies |

## Notes
- First launch of the `.exe` can take a few seconds while it unpacks.
- Scanned/image-only PDFs have no extractable text; add an OCR step if you need
  those.
