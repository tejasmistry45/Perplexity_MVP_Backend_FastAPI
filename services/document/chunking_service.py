import ticktoken
import re
from typing import List
from models.schemas import DocumentChunk
import logging

logger = logging.getLogger(__name__)

class DocumentChunkingService:
    def __init__(self):
        self.tokenizer = ticktoken.get_encoding("cl100k_base")
        self.max_chunk_size = 1000 # tokens
        self.overlap_size = 200    # tokens
        self.min_chunk_size = 200  # tokens

    def chunk_document(self, text: str, document_id: str) -> List[DocumentChunk]:
        """split document text into semantic chunks"""
        
        logger.info(f"Chunking Document: {document_id}")

        # split text by pages first
        pages = self._split_by_pages(text)
        all_chunks = []

        for page_num, page_text in enumerate(pages, 1):
            if not page_text.strip():
                continue

            page_chunk = self._chunk_page_text(
                text=page_text,
                page_number = page_num,
                document_id=document_id,
                start_index = len(all_chunks)
            )

        logger.info(f"Document {document_id} split into {len(all_chunks)} chunk")
        return all_chunks