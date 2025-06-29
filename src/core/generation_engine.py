"""
SmartDoc Analyzer - Generation Engine (Complete Version Compatible)
Handles response generation with both old and new OpenAI API versions
"""

import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class GenerationEngine:
    """Generates responses using OpenAI API with version compatibility"""
    
    def __init__(self):
        self.client = None
        self.model = "gpt-4o-mini"
        self.temperature = 0.2
        self.max_tokens = 2000
        self.api_version = None
        self._initialize_client()
        
        self.system_prompt = """You are an advanced document analysis assistant. 
        Provide comprehensive, accurate answers based on the retrieved context.
        Always cite sources and maintain factual accuracy. 
        Structure your responses clearly and professionally.
        
        When answering:
        1. Base your response primarily on the provided context
        2. Be specific and detailed when possible
        3. If the context doesn't contain enough information, say so
        4. Cite sources using the format [Source: filename/URL]
        """
    
    def _initialize_client(self):
        """Initialize OpenAI client with version detection"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found in environment variables")
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
                # Use compatible model for older API
                self.model = "gpt-3.5-turbo"
                logger.info("OpenAI client initialized successfully (v0.x)")
                return
            except ImportError:
                pass
            
            logger.error("No compatible OpenAI library found")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    async def generate_response(
        self, 
        query: str, 
        retrieval_results: List[Any],
        include_sources: bool = True,
        include_confidence: bool = True,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """Generate response based on query and retrieved context"""
        
        if not self.client:
            return self._generate_mock_response(query, retrieval_results, include_sources, include_confidence)
        
        try:
            # Prepare context from retrieval results
            context = self._prepare_context(retrieval_results)
            
            # Create the prompt
            user_prompt = self._create_prompt(query, context, detailed)
            
            # Generate response using appropriate API version
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
            
            # Calculate confidence score
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
        
        for i, result in enumerate(retrieval_results[:5], 1):  # Use top 5 results
            try:
                document = result.document
                content = document.page_content
                metadata = getattr(document, 'metadata', {})
                
                source = metadata.get('source', 'Unknown')
                chunk_id = metadata.get('chunk_id', '')
                title = metadata.get('title', '')
                
                # Truncate very long content
                if len(content) > 1000:
                    content = content[:1000] + "..."
                
                context_part = f"""
Context {i} (Relevance Score: {result.score:.3f}):
Source: {source} {f'(Chunk {chunk_id})' if chunk_id else ''}
{f'Title: {title}' if title else ''}
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
            detail_instruction = "\nProvide a detailed, comprehensive response with examples and explanations where appropriate."
        
        prompt = f"""
Based on the following context, please answer the user's question.

Context:
{context}

Question: {query}

Instructions:
- Answer based primarily on the provided context
- Be accurate and cite sources when possible using [Source: ...] format
- If the context doesn't contain sufficient information, acknowledge this
- Structure your response clearly with appropriate headings if needed{detail_instruction}
- Keep your response focused and relevant to the question

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
            min_score = min(scores)
            
            # Factor 1: Number of results (more results = more confidence)
            result_factor = min(len(retrieval_results) / 5.0, 1.0)
            
            # Factor 2: Average quality of top results
            score_factor = avg_score
            
            # Factor 3: Best result quality
            top_score_factor = max_score
            
            # Factor 4: Consistency of scores (low variance = higher confidence)
            if len(scores) > 1:
                variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
                consistency_factor = 1.0 / (1.0 + variance)
            else:
                consistency_factor = 1.0
            
            # Combine factors
            confidence = (
                result_factor * 0.25 + 
                score_factor * 0.35 + 
                top_score_factor * 0.25 + 
                consistency_factor * 0.15
            )
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5  # Default moderate confidence
    
    def _generate_mock_response(
        self, 
        query: str, 
        retrieval_results: List[Any],
        include_sources: bool,
        include_confidence: bool
    ) -> Dict[str, Any]:
        """Generate mock response when OpenAI API is not available"""
        
        if not retrieval_results:
            answer = """I couldn't find relevant information to answer your question. 

This could be because:
1. No documents have been processed yet
2. The query doesn't match any content in the processed documents
3. The document processing may have encountered errors

Please ensure documents are properly processed and try rephrasing your question."""
        else:
            # Create a simple extractive summary
            top_results = retrieval_results[:3]
            answer = "Based on the available documents, here's what I found:\n\n"
            
            for i, result in enumerate(top_results, 1):
                try:
                    content = result.document.page_content
                    source = getattr(result.document, 'metadata', {}).get('source', 'Unknown')
                    
                    # Extract most relevant sentences
                    sentences = content.split('.')[:3]  # First 3 sentences
                    excerpt = '. '.join(sentences) + '.'
                    
                    answer += f"**Finding {i}:**\n{excerpt}\n\n[Source: {source}]\n\n"
                    
                except Exception as e:
                    logger.warning(f"Error processing result {i}: {e}")
                    continue
            
            answer += "\n*Note: This is a simplified response as the AI language model is not currently available. For more sophisticated analysis, please configure your OpenAI API key.*"
        
        confidence = self._calculate_confidence(retrieval_results, query) if include_confidence else None
        
        return {
            'answer': answer,
            'confidence': confidence,
            'sources': retrieval_results if include_sources else None,
            'query': query,
            'model_used': 'mock_generator',
            'api_version': 'mock'
        }
    
    def _generate_fallback_response(
        self, 
        query: str, 
        retrieval_results: List[Any],
        include_sources: bool,
        include_confidence: bool
    ) -> Dict[str, Any]:
        """Generate fallback response when API call fails"""
        
        answer = """I encountered an error while generating a response. Here's a summary of the relevant information I found:

"""
        
        if retrieval_results:
            for i, result in enumerate(retrieval_results[:3], 1):
                try:
                    content = result.document.page_content
                    source = getattr(result.document, 'metadata', {}).get('source', 'Unknown')
                    
                    # Show first 200 characters of each result
                    excerpt = content[:200] + "..." if len(content) > 200 else content
                    answer += f"\n**Result {i}** (from {source}):\n{excerpt}\n"
                    
                except Exception as e:
                    logger.warning(f"Error processing fallback result {i}: {e}")
                    continue
        else:
            answer += "No relevant documents were found for your query."
        
        answer += "\n\n**Troubleshooting suggestions:**\n"
        answer += "1. Check your OpenAI API key configuration\n"
        answer += "2. Verify your internet connection\n"
        answer += "3. Try rephrasing your question\n"
        answer += "4. Ensure documents have been properly processed"
        
        return {
            'answer': answer,
            'confidence': 0.3 if include_confidence else None,
            'sources': retrieval_results if include_sources else None,
            'query': query,
            'model_used': 'fallback_generator',
            'api_version': self.api_version or 'unknown'
        }
    
    def update_settings(self, **kwargs):
        """Update generation settings"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"Updated generation setting {key}: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown setting: {key}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            'model': self.model,
            'api_version': self.api_version,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'client_available': self.client is not None
        }

# Test function
def test_generation():
    """Test function to verify generation engine is working"""
    try:
        engine = GenerationEngine()
        info = engine.get_model_info()
        
        print(f"✅ Generation engine test successful!")
        print(f"   - Model: {info['model']}")
        print(f"   - API version: {info['api_version'] or 'not available'}")
        print(f"   - Client available: {info['client_available']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation engine test failed: {e}")
        return False

if __name__ == "__main__":
    # Run test when module is executed directly
    test_generation()