import os
import asyncio

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

# Try to load Office support
try:
    from langchain_community.document_loaders import (
        UnstructuredWordDocumentLoader,
        UnstructuredExcelLoader,
        UnstructuredPowerPointLoader,
    )
    OFFICE_LOADER_AVAILABLE = True
except ImportError:
    OFFICE_LOADER_AVAILABLE = False
    print("⚠️ Office loaders not available, .doc/.xls/.ppt will be skipped.")

async def load_documents(folder_path: str):
    """
    Load documents from a folder asynchronously.
    Supports:
        - PDF TXT MD DOC / DOCX XLS / XLSX (if supported) PPT / PPTX (if supported)
    Returns:
        list[langchain.schema.Document]
    """

    all_docs = []
    filenames = os.listdir(folder_path)

    for filename in filenames:
        path = os.path.join(folder_path, filename)
        lower = filename.lower()

        loader = None

        # PDF
        if lower.endswith(".pdf"):
            loader = PyPDFLoader(path)

        # Text formats
        elif lower.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif lower.endswith(".md"):
            loader = TextLoader(path, encoding="utf-8")

        # Office formats (if available)
        elif OFFICE_LOADER_AVAILABLE and lower.endswith((".doc", ".docx")):
            loader = UnstructuredWordDocumentLoader(path)
        elif OFFICE_LOADER_AVAILABLE and lower.endswith((".xls", ".xlsx")):
            loader = UnstructuredExcelLoader(path)
        elif OFFICE_LOADER_AVAILABLE and lower.endswith((".ppt", ".pptx")):
            loader = UnstructuredPowerPointLoader(path)

        # Unsupported
        else:
            print(f"⏭️ Skipping unsupported file: {filename}")
            continue

        # Load documents using background thread, since loaders are blocking
        try:
            docs = await asyncio.to_thread(loader.load)
            print(f"📄 Loaded {len(docs)} docs from {filename}")
            all_docs.extend(docs)
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")
            continue

    print(f"✅ Total loaded documents: {len(all_docs)}")
    return all_docs
