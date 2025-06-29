"""
SmartDoc Analyzer - Retrieval Engine
Handles document retrieval and ranking
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    document: Any  # Document object
    score: float
    rank: int

class RetrievalEngine:
    """Advanced document retrieval with multiple strategies"""
    
    def __init__(self, vectorstore, embeddings, documents):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.documents = documents
        self.search_type = "hybrid"
        self.top_k = 5
        self.similarity_threshold = 0.7
        
    async def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        k = top_k or self.top_k
        
        try:
            if self.search_type == "semantic":
                return await self._semantic_search(query, k)
            elif self.search_type == "keyword":
                return await self._keyword_search(query, k)
            elif self.search_type == "hybrid":
                return await self._hybrid_search(query, k)
            else:
                logger.warning(f"Unknown search type: {self.search_type}, using semantic")
                return await self._semantic_search(query, k)
                
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
    
    async def _semantic_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Semantic search using vector similarity"""
        try:
            if not self.vectorstore:
                logger.warning("Vector store not available")
                return []
            
            # Use vector store similarity search
            results = self.vectorstore.similarity_search(query, k)
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for i, result in enumerate(results):
                # Create a simple document object
                doc = SimpleDocument(
                    page_content=result['text'],
                    metadata=result['metadata']
                )
                
                retrieval_results.append(RetrievalResult(
                    document=doc,
                    score=1.0 / (1.0 + result['score']),  # Convert distance to similarity
                    rank=i + 1
                ))
            
            logger.info(f"Semantic search returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def _keyword_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Keyword-based search using text matching"""
        try:
            if not self.documents:
                logger.warning("No documents available for keyword search")
                return []
            
            query_words = set(query.lower().split())
            scored_docs = []
            
            for doc in self.documents:
                text_words = set(doc.page_content.lower().split())
                
                # Simple TF-IDF-like scoring
                overlap = len(query_words.intersection(text_words))
                if overlap > 0:
                    score = overlap / len(query_words)
                    scored_docs.append((doc, score))
            
            # Sort by score and take top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            retrieval_results = []
            for i, (doc, score) in enumerate(scored_docs[:k]):
                retrieval_results.append(RetrievalResult(
                    document=doc,
                    score=score,
                    rank=i + 1
                ))
            
            logger.info(f"Keyword search returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    async def _hybrid_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Hybrid search combining semantic and keyword approaches"""
        try:
            # Get results from both methods
            semantic_results = await self._semantic_search(query, k * 2)
            keyword_results = await self._keyword_search(query, k * 2)
            
            # Combine and rerank results
            combined_scores = {}
            
            # Add semantic scores (weight: 0.7)
            for result in semantic_results:
                doc_id = self._get_doc_id(result.document)
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 0.7 * result.score
            
            # Add keyword scores (weight: 0.3)
            for result in keyword_results:
                doc_id = self._get_doc_id(result.document)
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + 0.3 * result.score
            
            # Create final results
            doc_map = {}
            for result in semantic_results + keyword_results:
                doc_id = self._get_doc_id(result.document)
                doc_map[doc_id] = result.document
            
            # Sort by combined score
            sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            
            retrieval_results = []
            for i, (doc_id, score) in enumerate(sorted_docs[:k]):
                retrieval_results.append(RetrievalResult(
                    document=doc_map[doc_id],
                    score=score,
                    rank=i + 1
                ))
            
            logger.info(f"Hybrid search returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to semantic search
            return await self._semantic_search(query, k)
    
    def _get_doc_id(self, document) -> str:
        """Generate a unique ID for a document"""
        try:
            # Use source and chunk_id if available
            metadata = getattr(document, 'metadata', {})
            source = metadata.get('source', 'unknown')
            chunk_id = metadata.get('chunk_id', 0)
            return f"{source}_{chunk_id}"
        except:
            # Fallback to hash of content
            content = getattr(document, 'page_content', str(document))
            return str(hash(content))
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval engine statistics"""
        return {
            'total_documents': len(self.documents) if self.documents else 0,
            'current_strategy': self.search_type,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'vectorstore_available': self.vectorstore is not None,
            'strategies_available': ['semantic', 'keyword', 'hybrid'],
            'reranking_enabled': True  # Always enabled in this implementation
        }
    
    def update_settings(self, **kwargs):
        """Update retrieval settings"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")

class SimpleDocument:
    """Simple document class for compatibility"""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def __str__(self):
        return self.page_content[:100] + "..." if len(self.page_content) > 100 else self.page_content

# Save this as src/core/retrieval_engine.py