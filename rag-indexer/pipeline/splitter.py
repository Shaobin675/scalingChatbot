from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Chunk_size: number of characters per chunk
        chunk_overlap: overlapping characters for context
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    async def split_text(self, text: str):
        # Split text asynchronously
        return await asyncio.to_thread(self.splitter.split_text, text)
