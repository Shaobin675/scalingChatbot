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

async def load_documents_from_file(path: str):
    """
    Load documents from a folder asynchronously.
    Supports:
        - PDF TXT MD DOC / DOCX XLS / XLSX (if supported) PPT / PPTX (if supported)
    Returns:
        list[langchain.schema.Document]
    """
    lower = path.lower()
    loader = None
    if lower.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        loader = TextLoader(path, encoding="utf-8")
    elif OFFICE_LOADER_AVAILABLE and lower.endswith((".doc", ".docx")):
        loader = UnstructuredWordDocumentLoader(path)
    elif OFFICE_LOADER_AVAILABLE and lower.endswith((".xls", ".xlsx")):
        loader = UnstructuredExcelLoader(path)
    elif OFFICE_LOADER_AVAILABLE and lower.endswith((".ppt", ".pptx")):
        loader = UnstructuredPowerPointLoader(path)
    else:
        return []

    docs = await asyncio.to_thread(loader.load)
    return docs
