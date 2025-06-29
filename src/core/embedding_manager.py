# ==================================================
# src/core/embedding_manager.py - COMPLETE VERSION
# ==================================================

"""
SmartDoc Analyzer - Embedding Manager (Complete Working Version)
Handles text embeddings with config integration and fallbacks
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embeddings with config integration"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.client = None
        self.model = "text-embedding-3-small"
        self.api_version = None
        self._initialize_client()
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment"""
        # Try config manager first
        if self.config_manager:
            try:
                api_config = self.config_manager.get_api_config()
                if api_config.openai_api_key:
                    return api_config.openai_api_key
            except Exception as e:
                logger.warning(f"Error getting API key from config: {e}")
        
        # Fallback to environment variable
        return os.getenv("OPENAI_API_KEY")
    
    def _initialize_client(self):
        """Initialize OpenAI client with version detection"""
        try:
            api_key = self._get_api_key()
            if not api_key:
                logger.warning("OpenAI API key not found in config or environment")
                return
            
            # Get model from config if available
            if self.config_manager:
                try:
                    api_config = self.config_manager.get_api_config()
                    self.model = api_config.embedding_model
                except Exception:
                    pass
            
            # Try new API first (v1.x)
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.api_version = "v1"
                logger.info("OpenAI client initialized successfully (v1.x)")
                return
            except ImportError:
                pass
            
            # Fall back to old API (v0.x)
            try:
                import openai
                openai.api_key = api_key
                self.client = openai
                self.api_version = "v0"
                self.model = "text-embedding-ada-002"
                logger.info("OpenAI client initialized successfully (v0.x)")
                return
            except ImportError:
                pass
            
            logger.error("No compatible OpenAI library found")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def get_embeddings(self):
        """Get embeddings instance for vector store"""
        if not self.client:
            logger.warning("OpenAI client not available, using mock embeddings")
            return MockEmbeddings()
        
        if self.api_version == "v1":
            return OpenAIEmbeddingsV1(client=self.client, model=self.model)
        elif self.api_version == "v0":
            return OpenAIEmbeddingsV0(client=self.client, model=self.model)
        else:
            return MockEmbeddings()

class OpenAIEmbeddingsV1:
    """OpenAI embeddings wrapper for v1.x API"""
    
    def __init__(self, client, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings using {self.model}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            return [[0.0] * 1536 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query"""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 1536

class OpenAIEmbeddingsV0:
    """OpenAI embeddings wrapper for v0.x API"""
    
    def __init__(self, client, model: str = "text-embedding-ada-002"):
        self.client = client
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.Embedding.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [embedding['embedding'] for embedding in response['data']]
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings using {self.model}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            return [[0.0] * 1536 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query"""
        try:
            response = self.client.Embedding.create(
                input=[text],
                model=self.model
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 1536

class MockEmbeddings:
    """Mock embeddings for when OpenAI API is not available"""
    
    def __init__(self):
        self.dimension = 1536
        logger.info("Using mock embeddings - OpenAI API not available")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return mock embeddings for documents"""
        import random
        import hashlib
        
        embeddings = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            random.seed(seed)
            embedding = [random.random() - 0.5 for _ in range(self.dimension)]
            embeddings.append(embedding)
        
        logger.info(f"Generated {len(embeddings)} mock embeddings")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Return mock embedding for query"""
        import random
        import hashlib
        
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        random.seed(seed)
        embedding = [random.random() - 0.5 for _ in range(self.dimension)]
        return embedding


# ==================================================
# src/core/generation_engine.py - COMPLETE VERSION
# ==================================================

"""
SmartDoc Analyzer - Generation Engine (Complete Working Version)
Handles response generation with config integration and fallbacks
"""

import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class GenerationEngine:
    """Generates responses using OpenAI API with config integration"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.client = None
        self.model = "gpt-4o-mini"
        self.temperature = 0.2
        self.max_tokens = 2000
        self.api_version = None
        self.system_prompt = """You are an advanced document analysis assistant. 
        Provide comprehensive, accurate answers based on the retrieved context.
        Always cite sources and maintain factual accuracy. 
        Structure your responses clearly and professionally."""
        self._load_config()
        self._initialize_client()
    
    def _load_config(self):
        """Load configuration settings"""
        if self.config_manager:
            try:
                api_config = self.config_manager.get_api_config()
                self.model = api_config.model
                self.temperature = api_config.temperature
                self.max_tokens = api_config.max_tokens
                
                gen_config = self.config_manager.get_generation_config()
                self.system_prompt = gen_config.system_prompt
                
            except Exception as e:
                logger.warning(f"Error loading config: {e}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from config or environment"""
        # Try config manager first
        if self.config_manager:
            try:
                api_config = self.config_manager.get_api_config()
                if api_config.openai_api_key:
                    return api_config.openai_api_key
            except Exception as e:
                logger.warning(f"Error getting API key from config: {e}")
        
        # Fallback to environment variable
        return os.getenv("OPENAI_API_KEY")
    
    def _initialize_client(self):
        """Initialize OpenAI client with version detection"""
        try:
            api_key = self._get_api_key()
            if not api_key:
                logger.warning("OpenAI API key not found in config or environment")
                return
            
            # Try new API first (v1.x)
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.api_version = "v1"
                logger.info("OpenAI client initialized successfully (v1.x)")
                return
            except ImportError:
                pass
            
            # Fall back to old API (v0.x)
            try:
                import openai
                openai.api_key = api_key
                self.client = openai
                self.api_version = "v0"
                self.model = "gpt-3.5-turbo"
                logger.info("OpenAI client initialized successfully (v0.x)")
                return
            except ImportError:
                pass
            
            logger.error("No compatible OpenAI library found")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    async def generate_response(self, query: str, retrieval_results: List[Any], 
                              include_sources: bool = True, include_confidence: bool = True, 
                              detailed: bool = False) -> Dict[str, Any]:
        """Generate response based on query and retrieved context"""
        
        if not self.client:
            return self._generate_mock_response(query, retrieval_results, include_sources, include_confidence)
        
        try:
            context = self._prepare_context(retrieval_results)
            user_prompt = self._create_prompt(query, context, detailed)
            
            if self.api_version == "v1":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                answer = response.choices[0].message.content
                
            elif self.api_version == "v0":
                response = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                answer = response['choices'][0]['message']['content']
            
            else:
                return self._generate_mock_response(query, retrieval_results, include_sources, include_confidence)
            
            confidence = self._calculate_confidence(retrieval_results, query)
            
            result = {
                'answer': answer,
                'confidence': confidence if include_confidence else None,
                'sources': retrieval_results if include_sources else None,
                'query': query,
                'model_used': self.model,
                'api_version': self.api_version
            }
            
            logger.info(f"Generated response for query: {query[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(query, retrieval_results, include_sources, include_confidence)
    
    def _prepare_context(self, retrieval_results: List[Any]) -> str:
        """Prepare context from retrieval results"""
        if not retrieval_results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(retrieval_results[:5], 1):
            try:
                document = result.document
                content = document.page_content
                metadata = getattr(document, 'metadata', {})
                
                source = metadata.get('source', 'Unknown')
                chunk_id = metadata.get('chunk_id', '')
                
                if len(content) > 1000:
                    content = content[:1000] + "..."
                
                context_part = f"""
Context {i} (Score: {result.score:.3f}):
Source: {source} {f'(Chunk {chunk_id})' if chunk_id else ''}
Content: {content}
---
"""
                context_parts.append(context_part)
                
            except Exception as e:
                logger.warning(f"Error processing retrieval result {i}: {e}")
                continue
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, detailed: bool = False) -> str:
        """Create the user prompt"""
        detail_instruction = ""
        if detailed:
            detail_instruction = "\nProvide a detailed, comprehensive response with examples and explanations."
        
        prompt = f"""
Based on the following context, please answer the user's question.

Context:
{context}

Question: {query}

Instructions:
- Answer based primarily on the provided context
- Be accurate and cite sources when possible
- If the context doesn't contain sufficient information, acknowledge this
- Structure your response clearly{detail_instruction}

Answer:"""
        
        return prompt
    
    def _calculate_confidence(self, retrieval_results: List[Any], query: str) -> float:
        """Calculate confidence score based on retrieval results"""
        if not retrieval_results:
            return 0.0
        
        try:
            scores = [result.score for result in retrieval_results[:3]]
            if not scores:
                return 0.0
            
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            
            result_factor = min(len(retrieval_results) / 5.0, 1.0)
            score_factor = avg_score
            top_score_factor = max_score
            
            confidence = (result_factor * 0.3 + score_factor * 0.4 + top_score_factor * 0.3)
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5
    
    def _generate_mock_response(self, query: str, retrieval_results: List[Any],
                              include_sources: bool, include_confidence: bool) -> Dict[str, Any]:
        """Generate mock response when OpenAI API is not available"""
        
        if not retrieval_results:
            answer = "I couldn't find relevant information to answer your question. Please ensure documents are properly processed and try again."
        else:
            top_result = retrieval_results[0]
            content = top_result.document.page_content
            source = getattr(top_result.document, 'metadata', {}).get('source', 'Unknown')
            
            answer = f"""Based on the available documents, here's what I found:

{content[:500]}{'...' if len(content) > 500 else ''}

[Source: {source}]

Note: This is a simplified response as the AI language model is not currently available."""
        
        confidence = self._calculate_confidence(retrieval_results, query) if include_confidence else None
        
        return {
            'answer': answer,
            'confidence': confidence,
            'sources': retrieval_results if include_sources else None,
            'query': query,
            'model_used': 'mock_generator'
        }
    
    def _generate_fallback_response(self, query: str, retrieval_results: List[Any],
                                  include_sources: bool, include_confidence: bool) -> Dict[str, Any]:
        """Generate fallback response when API call fails"""
        
        answer = """I encountered an error while generating a response. Here's a summary of the relevant information I found:

"""
        
        if retrieval_results:
            for i, result in enumerate(retrieval_results[:3], 1):
                try:
                    content = result.document.page_content
                    source = getattr(result.document, 'metadata', {}).get('source', 'Unknown')
                    answer += f"\n{i}. From {source}:\n{content[:200]}...\n"
                except:
                    continue
        else:
            answer += "No relevant documents were found for your query."
        
        answer += "\nPlease try rephrasing your question or check your API configuration."
        
        return {
            'answer': answer,
            'confidence': 0.3 if include_confidence else None,
            'sources': retrieval_results if include_sources else None,
            'query': query,
            'model_used': 'fallback_generator'
        }


# ==================================================
# SIMPLE PLACEHOLDER MODULES FOR OTHER COMPONENTS
# ==================================================

# src/core/document_processor.py
"""Simple Document Processor"""
import logging
from typing import List
logger = logging.getLogger(__name__)

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class DocumentProcessor:
    def __init__(self):
        logger.info("DocumentProcessor initialized")
    
    async def process_sources(self, sources: List[str]) -> List[Document]:
        logger.info(f"Processing {len(sources)} sources")
        return []

# src/core/vector_store_manager.py
"""Simple Vector Store Manager"""
import logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        logger.info("VectorStoreManager initialized")
    
    def create_vector_store(self, documents, embeddings):
        logger.info("Creating vector store")
        return None
    
    def load_vector_store(self, embeddings):
        logger.info("Loading vector store")
        return None
    
    def delete_vector_store(self):
        logger.info("Deleting vector store")

# src/core/retrieval_engine.py  
"""Simple Retrieval Engine"""
import logging
logger = logging.getLogger(__name__)

class RetrievalEngine:
    def __init__(self, vectorstore, embeddings, documents):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.documents = documents
        logger.info("RetrievalEngine initialized")
    
    async def retrieve(self, query: str):
        logger.info(f"Retrieving for query: {query}")
        return []
    
    def get_retrieval_stats(self):
        return {
            'total_documents': 0,
            'current_strategy': 'hybrid',
            'strategies_available': ['semantic', 'keyword', 'hybrid'],
            'reranking_enabled': True
        }