# ==================================================
# src/core/document_processor.py - FULLY FUNCTIONAL VERSION
# ==================================================

"""
SmartDoc Analyzer - Document Processor (Fully Functional)
Handles document loading, parsing, and chunking with real web scraping
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import requests
from urllib.parse import urlparse
import tempfile
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class Document:
    """Document representation with content and metadata"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def __str__(self):
        return f"Document(content_length={len(self.page_content)}, source={self.metadata.get('source', 'unknown')})"

class DocumentProcessor:
    """Document processing and chunking with real functionality"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.chunk_size = 1200
        self.chunk_overlap = 300
        self.separators = ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
        self._load_config()
        
    def _load_config(self):
        """Load configuration if available"""
        if self.config_manager:
            try:
                doc_config = self.config_manager.get_document_processing_config()
                self.chunk_size = doc_config.chunk_size
                self.chunk_overlap = doc_config.chunk_overlap
                self.separators = doc_config.separators
            except Exception as e:
                logger.warning(f"Could not load document processing config: {e}")
        
    async def process_sources(self, sources: List[str]) -> List[Document]:
        """Process multiple sources and return chunked documents"""
        all_documents = []
        
        logger.info(f"Starting to process {len(sources)} sources")
        
        for i, source in enumerate(sources, 1):
            try:
                logger.info(f"Processing source {i}/{len(sources)}: {source}")
                
                if self._is_url(source):
                    documents = await self._process_url(source)
                else:
                    documents = await self._process_file(source)
                
                all_documents.extend(documents)
                logger.info(f"Successfully processed {len(documents)} chunks from {source}")
                
            except Exception as e:
                logger.error(f"Failed to process {source}: {e}")
                continue
        
        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL"""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    async def _process_url(self, url: str) -> List[Document]:
        """Process a URL and extract text content"""
        try:
            # Enhanced web scraping with better headers and error handling
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            logger.info(f"Fetching URL: {url}")
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()
            
            # Basic content type check
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and 'text/plain' not in content_type:
                logger.warning(f"Unexpected content type: {content_type}")
            
            # Extract text content
            content = self._extract_text_from_html(response.text)
            
            if not content.strip():
                logger.warning(f"No text content extracted from {url}")
                return []
            
            # Create document with metadata
            title = self._extract_title(response.text)
            document = Document(
                page_content=content,
                metadata={
                    'source': url,
                    'source_type': 'url',
                    'title': title,
                    'content_type': content_type,
                    'processed_at': datetime.now().isoformat(),
                    'original_length': len(response.text),
                    'extracted_length': len(content)
                }
            )
            
            # Chunk the document
            chunks = self._chunk_document(document)
            logger.info(f"Extracted {len(content)} characters, created {len(chunks)} chunks")
            return chunks
            
        except requests.RequestException as e:
            logger.error(f"Network error processing URL {url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return []
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content"""
        try:
            # Try to use BeautifulSoup if available
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    script.decompose()
                
                # Get text and clean it up
                text = soup.get_text()
                
            except ImportError:
                logger.warning("BeautifulSoup not available, using regex-based extraction")
                # Fallback to regex-based extraction
                text = self._extract_text_regex(html_content)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return self._extract_text_regex(html_content)
    
    def _extract_text_regex(self, html_content: str) -> str:
        """Fallback regex-based text extraction"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common HTML entities
        html_entities = {
            '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
            '&quot;': '"', '&#39;': "'", '&hellip;': '...', '&mdash;': '—'
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        return text
    
    async def _process_file(self, file_path: str) -> List[Document]:
        """Process a file and extract text content"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return []
            
            # Read file based on extension
            if file_path.suffix.lower() in ['.txt', '.md']:
                content = self._read_text_file(file_path)
            elif file_path.suffix.lower() == '.pdf':
                content = self._read_pdf_file(file_path)
            elif file_path.suffix.lower() in ['.html', '.htm']:
                content = self._read_html_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return []
            
            if not content.strip():
                logger.warning(f"No content extracted from {file_path}")
                return []
            
            document = Document(
                page_content=content,
                metadata={
                    'source': str(file_path),
                    'source_type': 'file',
                    'filename': file_path.name,
                    'file_extension': file_path.suffix,
                    'file_size': file_path.stat().st_size,
                    'processed_at': datetime.now().isoformat(),
                    'content_length': len(content)
                }
            )
            
            chunks = self._chunk_document(document)
            logger.info(f"Processed file {file_path.name}: {len(content)} characters, {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def _read_text_file(self, file_path: Path) -> str:
        """Read plain text file"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file {file_path} with any supported encoding")
    
    def _read_pdf_file(self, file_path: Path) -> str:
        """Read PDF file (requires PyPDF2 or pdfplumber)"""
        try:
            # Try pdfplumber first (better for complex layouts)
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except ImportError:
                pass
            
            # Fallback to PyPDF2
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except ImportError:
                pass
            
            logger.warning("No PDF processing library available (install pdfplumber or PyPDF2)")
            return ""
            
        except Exception as e:
            logger.error(f"Error reading PDF file: {e}")
            return ""
    
    def _read_html_file(self, file_path: Path) -> str:
        """Read HTML file"""
        html_content = self._read_text_file(file_path)
        return self._extract_text_from_html(html_content)
    
    def _extract_title(self, html_content: str) -> str:
        """Extract title from HTML content"""
        try:
            # Try BeautifulSoup first
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                title_tag = soup.find('title')
                if title_tag:
                    return title_tag.get_text().strip()
            except ImportError:
                pass
            
            # Fallback to regex
            match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
            return "Untitled"
            
        except Exception as e:
            logger.warning(f"Error extracting title: {e}")
            return "Untitled"
    
    def _chunk_document(self, document: Document) -> List[Document]:
        """Split document into chunks using intelligent splitting"""
        text = document.page_content
        chunks = []
        
        if len(text) <= self.chunk_size:
            # Document is small enough, return as single chunk
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                'chunk_id': 0,
                'chunk_size': len(text),
                'start_index': 0,
                'end_index': len(text),
                'total_chunks': 1
            })
            
            chunks.append(Document(
                page_content=text,
                metadata=chunk_metadata
            ))
            return chunks
        
        # Split into chunks
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk_text = text[start:]
            else:
                # Find good split point
                chunk_text = text[start:end]
                
                # Try to split at sentence boundaries first
                for separator in self.separators:
                    split_point = chunk_text.rfind(separator)
                    if split_point > len(chunk_text) // 2:  # Don't make chunks too small
                        chunk_text = text[start:start + split_point + len(separator)]
                        break
            
            # Only add non-empty chunks
            if chunk_text.strip():
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_size': len(chunk_text),
                    'start_index': start,
                    'end_index': start + len(chunk_text),
                    'overlap_start': max(0, start - self.chunk_overlap),
                    'overlap_end': min(len(text), start + len(chunk_text) + self.chunk_overlap)
                })
                
                chunks.append(Document(
                    page_content=chunk_text.strip(),
                    metadata=chunk_metadata
                ))
                chunk_id += 1
            
            # Move to next chunk with overlap
            next_start = start + len(chunk_text) - self.chunk_overlap
            if next_start <= start:  # Prevent infinite loop
                next_start = start + max(1, len(chunk_text) // 2)
            
            start = next_start
            
            if start >= len(text):
                break
        
        # Update total chunks count in all chunk metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return ['.txt', '.md', '.html', '.htm', '.pdf']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'supported_formats': self.get_supported_formats(),
            'separators_count': len(self.separators)
        }


# ==================================================
# src/core/vector_store_manager.py - FUNCTIONAL VERSION
# ==================================================

"""
SmartDoc Analyzer - Vector Store Manager (Functional)
Handles vector database operations using FAISS with real functionality
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS library loaded successfully")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using mock vector store")

class VectorStoreManager:
    """Manages vector storage and retrieval using FAISS"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.storage_path = Path("data/vector_stores")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        if config_manager:
            try:
                vdb_config = config_manager.get_vector_db_config()
                self.storage_path = Path(vdb_config.storage_path)
                self.storage_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not load vector DB config: {e}")
        
    def create_vector_store(self, documents, embeddings):
        """Create a new vector store from documents"""
        if not documents:
            logger.error("No documents provided for vector store creation")
            return None
        
        try:
            logger.info(f"Creating vector store from {len(documents)} documents")
            
            if FAISS_AVAILABLE:
                return self._create_faiss_store(documents, embeddings)
            else:
                return self._create_mock_store(documents, embeddings)
                
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None
    
    def _create_faiss_store(self, documents, embeddings):
        """Create FAISS vector store"""
        logger.info("Creating FAISS vector store")
        
        # Extract text and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        logger.info(f"Generating embeddings for {len(texts)} documents")
        # Generate embeddings
        doc_embeddings = embeddings.embed_documents(texts)
        
        if not doc_embeddings or not doc_embeddings[0]:
            logger.error("Failed to generate embeddings")
            return None
        
        # Create FAISS index
        dimension = len(doc_embeddings[0])
        logger.info(f"Creating FAISS index with dimension {dimension}")
        
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        embeddings_array = np.array(doc_embeddings, dtype=np.float32)
        index.add(embeddings_array)
        
        # Create vector store object
        vector_store = FAISSVectorStore(
            index=index,
            texts=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            documents=documents
        )
        
        # Save to disk
        self._save_vector_store(vector_store)
        
        logger.info(f"Created FAISS vector store with {len(documents)} documents")
        return vector_store
    
    def _create_mock_store(self, documents, embeddings):
        """Create mock vector store for testing"""
        logger.info("Creating mock vector store")
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        vector_store = MockVectorStore(
            texts=texts,
            metadatas=metadatas,
            embeddings=embeddings,
            documents=documents
        )
        
        logger.info(f"Created mock vector store with {len(documents)} documents")
        return vector_store
    
    def load_vector_store(self, embeddings):
        """Load existing vector store from disk"""
        try:
            if FAISS_AVAILABLE:
                return self._load_faiss_store(embeddings)
            else:
                return self._load_mock_store(embeddings)
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None
    
    def _load_faiss_store(self, embeddings):
        """Load FAISS vector store from disk"""
        index_path = self.storage_path / "faiss_index.bin"
        metadata_path = self.storage_path / "metadata.pkl"
        
        if not (index_path.exists() and metadata_path.exists()):
            logger.warning("Vector store files not found")
            return None
        
        try:
            # Load FAISS index
            index = faiss.read_index(str(index_path))
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            
            vector_store = FAISSVectorStore(
                index=index,
                texts=data['texts'],
                metadatas=data['metadatas'],
                embeddings=embeddings,
                documents=data.get('documents', [])
            )
            
            logger.info("Loaded FAISS vector store from disk")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading FAISS store: {e}")
            return None
    
    def _load_mock_store(self, embeddings):
        """Load mock vector store"""
        metadata_path = self.storage_path / "mock_metadata.pkl"
        
        if not metadata_path.exists():
            logger.warning("Mock vector store not found")
            return None
        
        try:
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            
            vector_store = MockVectorStore(
                texts=data['texts'],
                metadatas=data['metadatas'],
                embeddings=embeddings,
                documents=data.get('documents', [])
            )
            
            logger.info("Loaded mock vector store from disk")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading mock store: {e}")
            return None
    
    def _save_vector_store(self, vector_store):
        """Save vector store to disk"""
        try:
            if isinstance(vector_store, FAISSVectorStore):
                self._save_faiss_store(vector_store)
            elif isinstance(vector_store, MockVectorStore):
                self._save_mock_store(vector_store)
                
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def _save_faiss_store(self, vector_store):
        """Save FAISS vector store to disk"""
        index_path = self.storage_path / "faiss_index.bin"
        metadata_path = self.storage_path / "metadata.pkl"
        
        # Save FAISS index
        faiss.write_index(vector_store.index, str(index_path))
        
        # Save metadata
        data = {
            'texts': vector_store.texts,
            'metadatas': vector_store.metadatas,
            'documents': getattr(vector_store, 'documents', [])
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info("Saved FAISS vector store to disk")
    
    def _save_mock_store(self, vector_store):
        """Save mock vector store to disk"""
        metadata_path = self.storage_path / "mock_metadata.pkl"
        
        data = {
            'texts': vector_store.texts,
            'metadatas': vector_store.metadatas,
            'documents': getattr(vector_store, 'documents', [])
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info("Saved mock vector store to disk")
    
    def delete_vector_store(self):
        """Delete stored vector store files"""
        try:
            for file_path in self.storage_path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            logger.info("Deleted vector store files")
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")

class FAISSVectorStore:
    """FAISS-based vector store with enhanced functionality"""
    
    def __init__(self, index, texts: List[str], metadatas: List[Dict], embeddings, documents: List = None):
        self.index = index
        self.texts = texts
        self.metadatas = metadatas
        self.embeddings = embeddings
        self.documents = documents or []
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search FAISS index
            scores, indices = self.index.search(query_vector, k)
            
            # Return results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if 0 <= idx < len(self.texts):  # Valid index
                    results.append({
                        'text': self.texts[idx],
                        'metadata': self.metadatas[idx],
                        'score': float(score),
                        'index': int(idx)
                    })
            
            logger.info(f"FAISS search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS similarity search: {e}")
            return []

class MockVectorStore:
    """Mock vector store for testing when FAISS is not available"""
    
    def __init__(self, texts: List[str], metadatas: List[Dict], embeddings, documents: List = None):
        self.texts = texts
        self.metadatas = metadatas
        self.embeddings = embeddings
        self.documents = documents or []
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Mock similarity search using simple text matching"""
        try:
            query_lower = query.lower()
            results = []
            
            for i, text in enumerate(self.texts):
                # Simple scoring based on keyword overlap
                text_lower = text.lower()
                score = 0
                
                # Count keyword matches
                for word in query_lower.split():
                    if len(word) > 2:  # Skip very short words
                        count = text_lower.count(word)
                        score += count * len(word)  # Weight by word length
                
                if score > 0:
                    # Normalize score by text length
                    normalized_score = score / len(text) * 1000
                    results.append({
                        'text': text,
                        'metadata': self.metadatas[i],
                        'score': normalized_score,
                        'index': i
                    })
            
            # Sort by score and return top k
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:k]
            
            logger.info(f"Mock search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in mock similarity search: {e}")
            return []


# ==================================================
# UPDATE TO app.py - PROCESSING FUNCTIONALITY
# ==================================================

# Add this method to your SmartDocAnalyzer class to replace the placeholder processing

async def process_documents_async(self, sources: List[str]) -> bool:
    """Process documents asynchronously with real functionality"""
    try:
        import streamlit as st
        
        # Update UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Document Processing
        status_text.text("🔄 Processing documents...")
        progress_bar.progress(25)
        
        chunks = await self.document_processor.process_sources(sources)
        if not chunks:
            st.error("❌ No documents could be processed")
            return False
        
        st.session_state.processed_documents = chunks
        
        # Step 2: Generate Embeddings
        status_text.text("🧠 Generating embeddings...")
        progress_bar.progress(50)
        
        embeddings = self.embedding_manager.get_embeddings()
        
        # Step 3: Create Vector Store
        status_text.text("🏗️ Building vector store...")
        progress_bar.progress(75)
        
        vectorstore = self.vector_store_manager.create_vector_store(chunks, embeddings)
        if not vectorstore:
            st.error("❌ Failed to create vector store")
            return False
        
        st.session_state.vectorstore = vectorstore
        
        # Step 4: Initialize Retrieval Engine
        status_text.text("🔍 Initializing retrieval engine...")
        progress_bar.progress(90)
        
        from src.core.retrieval_engine import RetrievalEngine
        retrieval_engine = RetrievalEngine(vectorstore, embeddings, chunks)
        st.session_state.retrieval_engine = retrieval_engine
        
        # Step 5: Complete
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        # Store processing statistics
        st.session_state.processing_stats = {
            "total_sources": len(sources),
            "total_chunks": len(chunks),
            "processing_time": datetime.now().isoformat(),
            "embedding_model": getattr(self.embedding_manager, 'model', 'unknown'),
            "chunk_size": self.document_processor.chunk_size
        }
        
        return True
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        st.error(f"❌ Processing failed: {str(e)}")
        return False