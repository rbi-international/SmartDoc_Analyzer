"""
SmartDoc Analyzer - Vector Store Manager
Handles vector database operations using FAISS
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS, use mock implementation if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using mock vector store")

class VectorStoreManager:
    """Manages vector storage and retrieval using FAISS"""
    
    def __init__(self):
        self.storage_path = Path("data/vector_stores")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    def create_vector_store(self, documents, embeddings):
        """Create a new vector store from documents"""
        if not documents:
            logger.error("No documents provided for vector store creation")
            return None
        
        try:
            if FAISS_AVAILABLE:
                return self._create_faiss_store(documents, embeddings)
            else:
                return self._create_mock_store(documents, embeddings)
                
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None
    
    def _create_faiss_store(self, documents, embeddings):
        """Create FAISS vector store"""
        # Extract text and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        doc_embeddings = embeddings.embed_documents(texts)
        
        # Create FAISS index
        dimension = len(doc_embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        embeddings_array = np.array(doc_embeddings, dtype=np.float32)
        index.add(embeddings_array)
        
        # Create vector store object
        vector_store = FAISSVectorStore(
            index=index,
            texts=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        # Save to disk
        self._save_vector_store(vector_store)
        
        logger.info(f"Created FAISS vector store with {len(documents)} documents")
        return vector_store
    
    def _create_mock_store(self, documents, embeddings):
        """Create mock vector store for testing"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        vector_store = MockVectorStore(
            texts=texts,
            metadatas=metadatas,
            embeddings=embeddings
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
        
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        vector_store = FAISSVectorStore(
            index=index,
            texts=data['texts'],
            metadatas=data['metadatas'],
            embeddings=embeddings
        )
        
        logger.info("Loaded FAISS vector store from disk")
        return vector_store
    
    def _load_mock_store(self, embeddings):
        """Load mock vector store"""
        metadata_path = self.storage_path / "mock_metadata.pkl"
        
        if not metadata_path.exists():
            logger.warning("Mock vector store not found")
            return None
        
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        vector_store = MockVectorStore(
            texts=data['texts'],
            metadatas=data['metadatas'],
            embeddings=embeddings
        )
        
        logger.info("Loaded mock vector store from disk")
        return vector_store
    
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
            'metadatas': vector_store.metadatas
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info("Saved FAISS vector store to disk")
    
    def _save_mock_store(self, vector_store):
        """Save mock vector store to disk"""
        metadata_path = self.storage_path / "mock_metadata.pkl"
        
        data = {
            'texts': vector_store.texts,
            'metadatas': vector_store.metadatas
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
    """FAISS-based vector store"""
    
    def __init__(self, index, texts: List[str], metadatas: List[Dict], embeddings):
        self.index = index
        self.texts = texts
        self.metadatas = metadatas
        self.embeddings = embeddings
    
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
                if idx < len(self.texts):  # Valid index
                    results.append({
                        'text': self.texts[idx],
                        'metadata': self.metadatas[idx],
                        'score': float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

class MockVectorStore:
    """Mock vector store for testing when FAISS is not available"""
    
    def __init__(self, texts: List[str], metadatas: List[Dict], embeddings):
        self.texts = texts
        self.metadatas = metadatas
        self.embeddings = embeddings
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Mock similarity search using simple text matching"""
        try:
            query_lower = query.lower()
            results = []
            
            for i, text in enumerate(self.texts):
                # Simple scoring based on keyword overlap
                text_lower = text.lower()
                score = 0
                
                for word in query_lower.split():
                    if word in text_lower:
                        score += text_lower.count(word)
                
                if score > 0:
                    results.append({
                        'text': text,
                        'metadata': self.metadatas[i],
                        'score': score
                    })
            
            # Sort by score and return top k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in mock similarity search: {e}")
            return []

# Save this as src/core/vector_store_manager.py