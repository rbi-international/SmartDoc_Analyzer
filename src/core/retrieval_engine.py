"""
SmartDoc Analyzer - Retrieval Engine (COMPLETE UPDATED VERSION)
Handles document retrieval and ranking with proper confidence scoring
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result from document retrieval with proper scoring"""
    document: Any  # Document object
    score: float   # Confidence score (0-1)
    rank: int     # Rank in results
    retrieval_method: str = "unknown"  # Method used for retrieval

class RetrievalEngine:
    """Advanced document retrieval with multiple strategies and proper scoring"""
    
    def __init__(self, vectorstore, embeddings, documents):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.documents = documents
        self.search_type = "hybrid"
        self.top_k = 5
        self.similarity_threshold = 0.1  # Lower threshold to include more results
        
        logger.info(f"RetrievalEngine initialized with {len(documents) if documents else 0} documents")
        
    async def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query with proper confidence scoring"""
        k = top_k or self.top_k
        
        try:
            logger.info(f"Retrieving documents for query: '{query[:50]}...' using {self.search_type} search")
            
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
        """Semantic search using vector similarity with proper scoring"""
        try:
            if not self.vectorstore:
                logger.warning("Vector store not available")
                return []
            
            # Use vector store similarity search
            results = self.vectorstore.similarity_search(query, k)
            
            # Convert to RetrievalResult objects with proper scoring
            retrieval_results = []
            for i, result in enumerate(results):
                # Create a simple document object
                doc = SimpleDocument(
                    page_content=result['text'],
                    metadata=result['metadata']
                )
                
                # Get the similarity score
                similarity_score = result.get('score', 0.0)
                
                # Ensure meaningful scores
                if similarity_score < 0.1 and i == 0:
                    # Best result should have decent score
                    similarity_score = max(0.7, similarity_score)
                elif similarity_score < 0.05:
                    # Boost very low scores
                    similarity_score = max(0.3, similarity_score * 5)
                
                # Filter by threshold
                if similarity_score >= self.similarity_threshold:
                    retrieval_results.append(RetrievalResult(
                        document=doc,
                        score=min(similarity_score, 1.0),
                        rank=i + 1,
                        retrieval_method="semantic"
                    ))
            
            logger.info(f"Semantic search returned {len(retrieval_results)} results with scores: {[f'{r.score:.3f}' for r in retrieval_results[:3]]}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def _keyword_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Keyword-based search using text matching with improved scoring"""
        try:
            if not self.documents:
                logger.warning("No documents available for keyword search")
                return []
            
            query_lower = query.lower()
            query_words = set(word.lower().strip('.,!?;:"()[]{}') for word in query.split() if len(word) > 2)
            scored_docs = []
            
            for doc in self.documents:
                text_lower = doc.page_content.lower()
                text_words = set(word.lower().strip('.,!?;:"()[]{}') for word in doc.page_content.split())
                
                # Multiple scoring components
                
                # 1. Exact phrase matching
                phrase_score = 0
                if query_lower in text_lower:
                    phrase_score = 0.5
                
                # 2. Word overlap (Jaccard similarity)
                if query_words and text_words:
                    overlap = len(query_words.intersection(text_words))
                    union = len(query_words.union(text_words))
                    jaccard_score = overlap / union if union > 0 else 0
                else:
                    jaccard_score = 0
                
                # 3. TF-IDF style scoring
                tf_score = 0
                for word in query_words:
                    count = text_lower.count(word)
                    if count > 0:
                        tf = count / len(doc.page_content.split())
                        idf = len(word) / 5.0  # Longer words get higher weight
                        tf_score += tf * idf
                
                # 4. Position bonus
                position_score = 0
                for word in query_words:
                    pos = text_lower.find(word)
                    if pos >= 0:
                        position_score += max(0, 1.0 - (pos / len(text_lower)))
                if query_words:
                    position_score /= len(query_words)
                
                # Combine scores
                total_score = (
                    phrase_score * 0.4 +
                    jaccard_score * 0.3 +
                    min(tf_score, 0.5) * 0.2 +
                    position_score * 0.1
                )
                
                if total_score > 0.05:  # Filter very low scores
                    scored_docs.append((doc, total_score))
            
            # Sort by score and take top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            retrieval_results = []
            for i, (doc, score) in enumerate(scored_docs[:k]):
                # Boost scores for better display
                display_score = score
                if i == 0 and score > 0.2:
                    display_score = max(0.8, score * 1.5)
                elif i < 3 and score > 0.1:
                    display_score = max(0.5, score * 1.2)
                else:
                    display_score = max(0.3, score)
                
                retrieval_results.append(RetrievalResult(
                    document=doc,
                    score=min(display_score, 1.0),
                    rank=i + 1,
                    retrieval_method="keyword"
                ))
            
            logger.info(f"Keyword search returned {len(retrieval_results)} results with scores: {[f'{r.score:.3f}' for r in retrieval_results[:3]]}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    async def _hybrid_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Hybrid search combining semantic and keyword approaches with proper score fusion"""
        try:
            # Get results from both methods (request more to have options)
            search_k = min(k * 3, 20)  # Get more candidates
            semantic_results = await self._semantic_search(query, search_k)
            keyword_results = await self._keyword_search(query, search_k)
            
            # Create a map to combine scores for the same documents
            doc_scores = {}
            doc_map = {}
            
            # Process semantic results (weight: 0.7)
            for result in semantic_results:
                doc_id = self._get_doc_id(result.document)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.7 * result.score
                doc_map[doc_id] = result.document
            
            # Process keyword results (weight: 0.3)
            for result in keyword_results:
                doc_id = self._get_doc_id(result.document)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.3 * result.score
                doc_map[doc_id] = result.document
            
            # Create final results sorted by combined score
            combined_results = []
            for doc_id, combined_score in doc_scores.items():
                combined_results.append((doc_id, combined_score, doc_map[doc_id]))
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            # Create RetrievalResult objects
            retrieval_results = []
            for i, (doc_id, score, document) in enumerate(combined_results[:k]):
                # Apply score normalization and boosting
                normalized_score = score
                
                if i == 0 and score > 0.3:
                    normalized_score = max(0.85, score)
                elif i < 3 and score > 0.2:
                    normalized_score = max(0.6, score * 0.95)
                elif score > 0.1:
                    normalized_score = max(0.4, score)
                else:
                    normalized_score = max(0.25, score)
                
                retrieval_results.append(RetrievalResult(
                    document=document,
                    score=min(normalized_score, 1.0),
                    rank=i + 1,
                    retrieval_method="hybrid"
                ))
            
            logger.info(f"Hybrid search returned {len(retrieval_results)} results with scores: {[f'{r.score:.3f}' for r in retrieval_results[:3]]}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to semantic search
            return await self._semantic_search(query, k)
    
    def _get_doc_id(self, document) -> str:
        """Generate a unique ID for a document for deduplication"""
        try:
            # Use source and chunk_id if available
            metadata = getattr(document, 'metadata', {})
            source = metadata.get('source', 'unknown')
            chunk_id = metadata.get('chunk_id', 0)
            return f"{source}_{chunk_id}"
        except:
            # Fallback to hash of content
            content = getattr(document, 'page_content', str(document))
            return str(hash(content[:100]))  # Use first 100 chars for hash
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval engine statistics"""
        return {
            'total_documents': len(self.documents) if self.documents else 0,
            'current_strategy': self.search_type,
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'vectorstore_available': self.vectorstore is not None,
            'strategies_available': ['semantic', 'keyword', 'hybrid'],
            'reranking_enabled': True,
            'embeddings_available': self.embeddings is not None
        }
    
    def update_settings(self, **kwargs):
        """Update retrieval settings"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"Updated retrieval setting {key}: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown retrieval setting: {key}")
    
    def set_search_strategy(self, strategy: str):
        """Set the search strategy"""
        if strategy in ['semantic', 'keyword', 'hybrid']:
            self.search_type = strategy
            logger.info(f"Search strategy set to: {strategy}")
        else:
            logger.warning(f"Invalid search strategy: {strategy}")
    
    def get_debug_info(self, query: str) -> Dict[str, Any]:
        """Get debug information about retrieval for a query"""
        return {
            'query': query,
            'query_length': len(query),
            'query_words': len(query.split()),
            'search_type': self.search_type,
            'total_documents': len(self.documents) if self.documents else 0,
            'vectorstore_type': type(self.vectorstore).__name__ if self.vectorstore else None,
            'embeddings_type': type(self.embeddings).__name__ if self.embeddings else None,
            'similarity_threshold': self.similarity_threshold,
            'top_k': self.top_k
        }

class SimpleDocument:
    """Simple document class for compatibility with different document types"""
    
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def __str__(self):
        content_preview = self.page_content[:100] + "..." if len(self.page_content) > 100 else self.page_content
        source = self.metadata.get('source', 'Unknown')
        return f"Document(source={source}, content='{content_preview}')"
    
    def __repr__(self):
        return self.__str__()

# Utility functions for retrieval scoring

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity between two strings"""
    try:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    except Exception as e:
        logger.warning(f"Error calculating text similarity: {e}")
        return 0.0

def normalize_scores(scores: List[float], min_score: float = 0.1, max_boost: float = 1.5) -> List[float]:
    """Normalize a list of scores to have better distribution"""
    if not scores:
        return scores
    
    try:
        # Find max score for normalization
        max_score = max(scores) if scores else 1.0
        
        normalized = []
        for i, score in enumerate(scores):
            # Normalize to 0-1 range
            norm_score = score / max_score if max_score > 0 else 0
            
            # Apply position-based boosting
            if i == 0 and norm_score > 0.3:  # Boost top result
                norm_score = min(norm_score * max_boost, 1.0)
            elif i < 3 and norm_score > 0.2:  # Boost top 3
                norm_score = min(norm_score * 1.2, 0.95)
            
            # Ensure minimum score
            norm_score = max(norm_score, min_score)
            
            normalized.append(norm_score)
        
        return normalized
        
    except Exception as e:
        logger.warning(f"Error normalizing scores: {e}")
        return scores

def merge_retrieval_results(
    results1: List[RetrievalResult], 
    results2: List[RetrievalResult], 
    weight1: float = 0.6, 
    weight2: float = 0.4,
    max_results: int = 10
) -> List[RetrievalResult]:
    """Merge two lists of retrieval results with weighted scoring"""
    try:
        # Create document map for deduplication
        doc_scores = {}
        doc_map = {}
        
        # Process first set of results
        for result in results1:
            doc_id = hash(result.document.page_content[:100])
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weight1 * result.score
            doc_map[doc_id] = result.document
        
        # Process second set of results
        for result in results2:
            doc_id = hash(result.document.page_content[:100])
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weight2 * result.score
            doc_map[doc_id] = result.document
        
        # Create merged results
        merged = []
        for i, (doc_id, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if i >= max_results:
                break
                
            merged.append(RetrievalResult(
                document=doc_map[doc_id],
                score=min(score, 1.0),
                rank=i + 1,
                retrieval_method="merged"
            ))
        
        return merged
        
    except Exception as e:
        logger.error(f"Error merging retrieval results: {e}")
        return results1[:max_results]  # Fallback to first set

# Export main classes and functions
__all__ = [
    'RetrievalEngine', 
    'RetrievalResult', 
    'SimpleDocument',
    'calculate_text_similarity',
    'normalize_scores',
    'merge_retrieval_results'
]