"""
SmartDoc Analyzer - Main Application (COMPLETE FUNCTIONAL VERSION)
Advanced Document Intelligence Research Platform
"""

import asyncio
import streamlit as st
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# Create logs directory before setting up logging
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging with proper error handling
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/smartdoc.log'),
            logging.StreamHandler()
        ]
    )
except Exception as e:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    print(f"Warning: Could not set up file logging: {e}")

logger = logging.getLogger(__name__)

# Import modules with error handling
try:
    from src.config.config_manager import get_config_manager, reload_config
    from src.core.document_processor import DocumentProcessor
    from src.core.vector_store_manager import VectorStoreManager
    from src.core.retrieval_engine import RetrievalEngine
    from src.core.embedding_manager import EmbeddingManager
    from src.core.generation_engine import GenerationEngine
    COMPONENTS_LOADED = True
    logger.info("All components loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    COMPONENTS_LOADED = False
    class DummyClass:
        def __init__(self, *args, **kwargs):
            pass
    DocumentProcessor = DummyClass
    VectorStoreManager = DummyClass
    RetrievalEngine = DummyClass
    EmbeddingManager = DummyClass
    GenerationEngine = DummyClass

class SmartDocAnalyzer:
    """Main application class for SmartDoc Analyzer"""
    
    def __init__(self):
        self.config_manager = None
        self.document_processor = None
        self.vector_store_manager = None
        self.embedding_manager = None
        self.generation_engine = None
        
        if COMPONENTS_LOADED:
            try:
                self.config_manager = get_config_manager()
                self.config_manager.ensure_directories()
                
                self.document_processor = DocumentProcessor(config_manager=self.config_manager)
                self.vector_store_manager = VectorStoreManager(config_manager=self.config_manager)
                
                try:
                    self.embedding_manager = EmbeddingManager(config_manager=self.config_manager)
                    logger.info("EmbeddingManager initialized with config integration")
                except TypeError:
                    self.embedding_manager = EmbeddingManager()
                    logger.warning("Using old EmbeddingManager without config integration")
                
                try:
                    self.generation_engine = GenerationEngine(config_manager=self.config_manager)
                    logger.info("GenerationEngine initialized with config integration")
                except TypeError:
                    self.generation_engine = GenerationEngine()
                    logger.warning("Using old GenerationEngine without config integration")
                
                self._initialize_session_state()
                logger.info("SmartDoc Analyzer initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize SmartDocAnalyzer: {e}")
                self._create_minimal_config()
        else:
            self._create_minimal_config()
    
    def _create_minimal_config(self):
        """Create minimal configuration for when full config fails"""
        class MinimalConfigManager:
            def ensure_directories(self):
                for dir_name in ['data/vector_stores', 'data/uploads', 'data/cache', 'logs']:
                    Path(dir_name).mkdir(parents=True, exist_ok=True)
            
            def get_interface_config(self):
                class InterfaceConfig:
                    title = "SmartDoc Analyzer"
                    subtitle = "Advanced Document Intelligence Research Platform"
                    layout = "wide"
                    supported_formats = ["txt", "pdf", "docx", "html", "md"]
                return InterfaceConfig()
            
            def validate_api_key(self):
                env_key = os.getenv("OPENAI_API_KEY")
                if env_key:
                    return True
                
                try:
                    import yaml
                    config_file = Path("config.yaml")
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config_data = yaml.safe_load(f)
                        api_key = config_data.get('api', {}).get('openai', {}).get('api_key', '')
                        return bool(api_key and len(api_key) > 20)
                except Exception as e:
                    logger.warning(f"Error checking config file: {e}")
                
                return False
            
            def get_nested_value(self, path, default=None):
                try:
                    import yaml
                    config_file = Path("config.yaml")
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config_data = yaml.safe_load(f)
                        
                        keys = path.split('.')
                        value = config_data
                        for key in keys:
                            value = value[key]
                        return value
                except Exception:
                    pass
                
                fallbacks = {
                    "retrieval.search_type": "hybrid",
                    "retrieval.top_k": 5,
                    "retrieval.similarity_threshold": 0.7,
                    "api.openai.model": "gpt-4o-mini",
                    "api.openai.embedding_model": "text-embedding-3-small",
                    "document_processing.chunk_size": 1200,
                    "document_processing.chunk_overlap": 300,
                    "retrieval.reranking_enabled": True
                }
                return fallbacks.get(path, default)
            
            def update_config(self, path, value):
                logger.info(f"Config update (minimal mode): {path} = {value}")
        
        self.config_manager = MinimalConfigManager()
        
        if not COMPONENTS_LOADED:
            self.document_processor = None
            self.vector_store_manager = None
            self.embedding_manager = None
            self.generation_engine = None
        
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = []
        if 'retrieval_engine' not in st.session_state:
            st.session_state.retrieval_engine = None
        if 'processing_stats' not in st.session_state:
            st.session_state.processing_stats = {}
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
    
    def configure_page(self):
        """Configure Streamlit page"""
        try:
            interface_config = self.config_manager.get_interface_config()
            st.set_page_config(
                page_title=interface_config.title,
                page_icon="🔬",
                layout=interface_config.layout,
                initial_sidebar_state="expanded"
            )
        except Exception as e:
            logger.error(f"Error configuring page: {e}")
            st.set_page_config(
                page_title="SmartDoc Analyzer",
                page_icon="🔬",
                layout="wide",
                initial_sidebar_state="expanded"
            )
        
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active { background-color: #28a745; }
        .status-inactive { background-color: #dc3545; }
        .status-warning { background-color: #ffc107; }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render application header"""
        try:
            interface_config = self.config_manager.get_interface_config()
            title = interface_config.title
            subtitle = interface_config.subtitle
        except:
            title = "SmartDoc Analyzer"
            subtitle = "Advanced Document Intelligence Research Platform"
        
        st.markdown(f"""
        <div class="main-header">
            <h1>🔬 {title}</h1>
            <p>{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls and status"""
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            missing_components = self._check_missing_components()
            if missing_components:
                self._render_missing_components_warning(missing_components)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ All Components Loaded</h4>
                    <p>SmartDoc Analyzer is ready to use!</p>
                </div>
                """, unsafe_allow_html=True)
            
            self._render_system_status()
            self._render_configuration_section()
            
            if not missing_components:
                self._render_data_input_section()
                self._render_processing_controls()
            else:
                self._render_disabled_data_input()
            
            self._render_advanced_options()
    
    def _check_missing_components(self):
        """Check for missing components"""
        missing = []
        if not COMPONENTS_LOADED:
            missing.append("Core modules not imported")
        if self.document_processor is None:
            missing.append("Document Processor")
        if self.vector_store_manager is None:
            missing.append("Vector Store Manager")
        if self.embedding_manager is None:
            missing.append("Embedding Manager")
        if self.generation_engine is None:
            missing.append("Generation Engine")
        return missing
    
    def _render_missing_components_warning(self, missing_components):
        """Render warning about missing components"""
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ Missing Components</h4>
            <p>The following components could not be loaded:</p>
            <ul>
                {''.join(f'<li>{comp}</li>' for comp in missing_components)}
            </ul>
            <p><strong>Fix steps:</strong></p>
            <ol>
                <li>Create __init__.py files in src/, src/config/, src/core/</li>
                <li>Ensure all core modules exist in src/core/</li>
                <li>Restart the application</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_system_status(self):
        """Render system status indicators"""
        st.subheader("📊 System Status")
        
        try:
            api_status = "active" if self.config_manager.validate_api_key() else "inactive"
            st.markdown(f"""
            <div class="metric-card">
                <span class="status-indicator status-{api_status}"></span>
                <strong>API Connection:</strong> {'Connected' if api_status == 'active' else 'Disconnected'}
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown("""
            <div class="metric-card">
                <span class="status-indicator status-warning"></span>
                <strong>API Connection:</strong> Error checking
            </div>
            """, unsafe_allow_html=True)
        
        vs_status = "active" if st.session_state.vectorstore else "inactive"
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator status-{vs_status}"></span>
            <strong>Vector Store:</strong> {'Ready' if vs_status == 'active' else 'Not Loaded'}
        </div>
        """, unsafe_allow_html=True)
        
        doc_count = len(st.session_state.processed_documents)
        st.markdown(f"""
        <div class="metric-card">
            <strong>Documents Processed:</strong> {doc_count}
        </div>
        """, unsafe_allow_html=True)
        
        components_ok = not bool(self._check_missing_components())
        comp_status = "active" if components_ok else "inactive"
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator status-{comp_status}"></span>
            <strong>Core Components:</strong> {'Loaded' if components_ok else 'Missing'}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_configuration_section(self):
        """Render configuration options"""
        with st.expander("⚙️ Configuration", expanded=False):
            st.subheader("🔍 Retrieval Settings")
            
            try:
                current_strategy = self.config_manager.get_nested_value("retrieval.search_type", "hybrid")
                search_strategy = st.selectbox(
                    "Search Strategy",
                    ["semantic", "keyword", "hybrid"],
                    index=["semantic", "keyword", "hybrid"].index(current_strategy)
                )
                
                top_k = st.slider(
                    "Number of Results",
                    min_value=1,
                    max_value=20,
                    value=self.config_manager.get_nested_value("retrieval.top_k", 5)
                )
                
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.config_manager.get_nested_value("retrieval.similarity_threshold", 0.7),
                    step=0.05
                )
                
                if st.button("💾 Save Configuration"):
                    try:
                        self.config_manager.update_config("retrieval.search_type", search_strategy)
                        self.config_manager.update_config("retrieval.top_k", top_k)
                        self.config_manager.update_config("retrieval.similarity_threshold", similarity_threshold)
                        st.success("Configuration saved!")
                    except Exception as e:
                        st.error(f"Failed to save configuration: {e}")
            except Exception as e:
                st.error(f"Configuration error: {e}")
    
    def _render_data_input_section(self):
        """Render data input options when components are loaded"""
        st.subheader("📁 Data Input")
        
        tab1, tab2, tab3 = st.tabs(["URLs", "File Upload", "Batch Upload"])
        
        with tab1:
            url_text = st.text_area(
                "Enter URLs (one per line):",
                height=100,
                placeholder="https://example.com/document1\nhttps://example.com/document2"
            )
            urls = [url.strip() for url in url_text.split('\n') if url.strip()] if url_text else []
            
            if urls:
                st.info(f"📋 {len(urls)} URLs ready for processing")
                for i, url in enumerate(urls[:5], 1):
                    st.write(f"{i}. {url}")
                if len(urls) > 5:
                    st.write(f"... and {len(urls) - 5} more")
        
        with tab2:
            st.info("File upload feature coming soon!")
            st.text_area("Placeholder for file upload", disabled=True, height=100)
        
        with tab3:
            st.info("Batch upload feature coming soon!")
            st.text_area("Placeholder for batch upload", disabled=True, height=100)
        
        st.session_state.current_sources = urls
        return urls
    
    def _render_disabled_data_input(self):
        """Render disabled data input when components are missing"""
        st.subheader("📁 Data Input")
        st.info("🚧 Document processing requires all core components to be loaded. Please fix the missing components first.")
        st.text_area(
            "Enter URLs (one per line):",
            height=100,
            placeholder="Please fix missing components first...",
            disabled=True
        )
    
    def _render_processing_controls(self):
        """Render processing control buttons with real functionality"""
        st.subheader("🔄 Processing Controls")
        
        sources = getattr(st.session_state, 'current_sources', [])
        components_ok = not bool(self._check_missing_components())
        
        try:
            api_ok = self.config_manager.validate_api_key()
        except:
            api_ok = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(
                "🚀 Process Documents",
                disabled=not (sources and components_ok and api_ok),
                help="Process all input sources" if components_ok and api_ok else "Requires components and API key"
            ):
                st.session_state.process_button_clicked = True
        
        with col2:
            if st.button(
                "📂 Load Existing",
                disabled=not (components_ok and api_ok),
                help="Load previously processed documents" if components_ok and api_ok else "Requires components and API key"
            ):
                st.session_state.load_button_clicked = True
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.vectorstore = None
            st.session_state.processed_documents = []
            st.session_state.retrieval_engine = None
            st.session_state.processing_stats = {}
            st.session_state.query_history = []
            st.success("All data cleared!")
            st.rerun()
    
    def _render_advanced_options(self):
        """Render advanced options"""
        with st.expander("🧪 Advanced Options", expanded=False):
            components_ok = not bool(self._check_missing_components())
            
            if components_ok:
                st.success("✅ Advanced options are available!")
                
                st.subheader("🤖 Model Settings")
                st.selectbox("Language Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
                st.selectbox("Embedding Model", ["text-embedding-3-small", "text-embedding-3-large"])
                
                st.subheader("📄 Processing Options")
                st.slider("Chunk Size", 500, 2000, 1200, 100)
                st.slider("Chunk Overlap", 0, 500, 300, 50)
                
                st.subheader("🧬 Experimental Features")
                st.checkbox("Enable Neural Reranking", value=True)
                st.checkbox("Extract Enhanced Metadata", value=True)
                
                if st.button("🔧 Apply Advanced Settings"):
                    st.success("Advanced settings would be applied (feature coming soon)!")
                
            else:
                st.info("Advanced options will be available once all components are loaded.")
    
    async def process_documents_async(self, sources: List[str]) -> bool:
        """Process documents asynchronously with real functionality"""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Processing documents...")
            progress_bar.progress(25)
            
            chunks = await self.document_processor.process_sources(sources)
            if not chunks:
                st.error("❌ No documents could be processed")
                return False
            
            st.session_state.processed_documents = chunks
            
            status_text.text("🧠 Generating embeddings...")
            progress_bar.progress(50)
            
            embeddings = self.embedding_manager.get_embeddings()
            
            status_text.text("🏗️ Building vector store...")
            progress_bar.progress(75)
            
            vectorstore = self.vector_store_manager.create_vector_store(chunks, embeddings)
            if not vectorstore:
                st.error("❌ Failed to create vector store")
                return False
            
            st.session_state.vectorstore = vectorstore
            
            status_text.text("🔍 Initializing retrieval engine...")
            progress_bar.progress(90)
            
            retrieval_engine = RetrievalEngine(vectorstore, embeddings, chunks)
            st.session_state.retrieval_engine = retrieval_engine
            
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            
            st.session_state.processing_stats = {
                "total_sources": len(sources),
                "total_chunks": len(chunks),
                "processing_time": datetime.now().isoformat(),
                "embedding_model": getattr(self.embedding_manager, 'model', 'unknown'),
                "chunk_size": getattr(self.document_processor, 'chunk_size', 'unknown')
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            st.error(f"❌ Processing failed: {str(e)}")
            return False
    
    def load_existing_data(self) -> bool:
        """Load existing processed data"""
        try:
            embeddings = self.embedding_manager.get_embeddings()
            vectorstore = self.vector_store_manager.load_vector_store(embeddings)
            
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                retrieval_engine = RetrievalEngine(vectorstore, embeddings, [])
                st.session_state.retrieval_engine = retrieval_engine
                st.success("✅ Existing data loaded successfully!")
                return True
            else:
                st.warning("⚠️ No existing data found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
            st.error(f"❌ Loading failed: {str(e)}")
            return False
    
    def render_main_content(self):
        """Render main content area"""
        missing_components = self._check_missing_components()
        
        if missing_components:
            self._render_setup_screen(missing_components)
        elif st.session_state.vectorstore is None:
            self._render_welcome_screen()
        else:
            self._render_analysis_interface()
    
    def _render_setup_screen(self, missing_components):
        """Render setup screen when components are missing"""
        st.markdown("## 🔧 Setup Required")
        st.markdown("SmartDoc Analyzer needs some components to be set up before it can run properly.")
        
        st.markdown("### Missing Components:")
        for comp in missing_components:
            st.markdown(f"- ❌ {comp}")
        
        st.markdown("""
        ### Quick Fix Steps
        
        **1. Create missing __init__.py files:**
        ```bash
        touch src/__init__.py
        touch src/config/__init__.py
        touch src/core/__init__.py
        ```
        
        **2. Ensure all core modules exist in src/core/:**
        - document_processor.py
        - vector_store_manager.py
        - retrieval_engine.py
        - embedding_manager.py
        - generation_engine.py
        
        **3. Restart the application:**
        ```bash
        streamlit run app.py
        ```
        """)
    
    def _render_welcome_screen(self):
        """Render welcome screen when no data is loaded"""
        try:
            api_status = "✅ Connected" if self.config_manager.validate_api_key() else "❌ Not Connected"
        except:
            api_status = "❌ Not Connected"
        
        st.markdown(f"""
        ## 🚀 Welcome to SmartDoc Analyzer
        
        ### Current Status
        
        **API Connection:** {api_status}  
        **Components:** ✅ All Loaded  
        **Configuration:** ✅ Loaded from config.yaml  
        
        ### Getting Started
        
        1. **Add URLs** in the sidebar Data Input section
        2. **Click Process Documents** to build your knowledge base  
        3. **Ask questions** and get intelligent answers
        
        ### Features
        
        - 🔍 **Advanced Retrieval**: Semantic, keyword, and hybrid search
        - 🧠 **Neural Reranking**: State-of-the-art result ranking
        - 📊 **Rich Analytics**: Detailed insights into your data
        - 🔧 **Configurable**: Flexible settings for different use cases
        
        ### Ready to Start!
        
        Your SmartDoc Analyzer is fully configured and ready to process documents. 
        Add some URLs in the sidebar to get started!
        """)
    
    def _render_analysis_interface(self):
        """Render main analysis interface with real functionality"""
        st.markdown("## 📊 Document Analysis Dashboard")
        st.markdown("Your documents have been processed and are ready for analysis!")
        
        if st.session_state.processing_stats:
            col1, col2, col3, col4 = st.columns(4)
            stats = st.session_state.processing_stats
            
            with col1:
                st.metric("Sources Processed", stats.get('total_sources', 0))
            with col2:
                st.metric("Document Chunks", stats.get('total_chunks', 0))
            with col3:
                st.metric("Embedding Model", stats.get('embedding_model', 'Unknown'))
            with col4:
                st.metric("Chunk Size", stats.get('chunk_size', 0))
        
        tab1, tab2, tab3 = st.tabs(["💬 Query Interface", "📊 Analytics", "🔍 Document Explorer"])
        
        with tab1:
            self._render_query_interface()
        with tab2:
            self._render_analytics_dashboard()
        with tab3:
            self._render_document_explorer()
    
    def _render_query_interface(self):
        """Render query interface for asking questions"""
        st.header("💬 Ask Questions About Your Documents")
        
        st.subheader("🎯 Quick Queries")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📋 Summarize All"):
                st.session_state.current_query = "Provide a comprehensive summary of all the documents."
        
        with col2:
            if st.button("🔑 Key Points"):
                st.session_state.current_query = "What are the main key points and findings?"
        
        with col3:
            if st.button("📈 Main Topics"):
                st.session_state.current_query = "What are the main topics discussed in the documents?"
        
        with col4:
            if st.button("💡 Insights"):
                st.session_state.current_query = "What are the most important insights from the content?"
        
        query = st.text_area(
            "Enter your question:",
            value=getattr(st.session_state, 'current_query', ''),
            height=100,
            placeholder="Ask anything about your processed documents..."
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            include_sources = st.checkbox("Include sources", value=True)
        with col2:
            include_confidence = st.checkbox("Show confidence", value=True)
        with col3:
            detailed_response = st.checkbox("Detailed response", value=False)
        
        if query and st.button("🔍 Analyze Query", type="primary"):
            if st.session_state.retrieval_engine:
                with st.spinner("🤔 Analyzing your question..."):
                    try:
                        vectorstore = st.session_state.vectorstore
                        results = vectorstore.similarity_search(query, k=5)
                        
                        if results:
                            st.subheader("📋 Analysis Results")
                            
                            for i, result in enumerate(results[:3], 1):
                                with st.expander(f"📄 Result {i} (Score: {result['score']:.3f})"):
                                    st.write("**Content:**")
                                    content = result['text']
                                    st.write(content[:500] + "..." if len(content) > 500 else content)
                                    
                                    if include_sources:
                                        st.write("**Source:**")
                                        metadata = result['metadata']
                                        source = metadata.get('source', 'Unknown')
                                        st.write(f"- {source}")
                                        
                                        if include_confidence:
                                            st.write(f"**Relevance Score:** {result['score']:.3f}")
                        else:
                            st.warning("⚠️ No relevant documents found for your query")
                            
                    except Exception as e:
                        st.error(f"❌ Query processing failed: {str(e)}")
            else:
                st.error("❌ No retrieval engine available. Please process documents first.")
    
    def _render_analytics_dashboard(self):
        """Render analytics dashboard"""
        st.header("📊 Document Analytics")
        
        if not st.session_state.processed_documents:
            st.info("📋 No documents processed yet")
            return
        
        docs = st.session_state.processed_documents
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", len(docs))
        
        with col2:
            total_chars = sum(len(doc.page_content) for doc in docs)
            st.metric("Total Characters", f"{total_chars:,}")
        
        with col3:
            avg_length = total_chars / len(docs) if docs else 0
            st.metric("Avg. Document Length", f"{avg_length:.0f}")
        
        with col4:
            unique_sources = len(set(doc.metadata.get('source', 'unknown') for doc in docs))
            st.metric("Unique Sources", unique_sources)
        
        st.subheader("📏 Document Length Distribution")
        
        lengths = [len(doc.page_content) for doc in docs]
        df_lengths = pd.DataFrame({'Length': lengths, 'Document': range(len(lengths))})
        
        fig = px.histogram(df_lengths, x='Length', nbins=20, title="Distribution of Document Lengths")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔗 Source Analysis")
        
        source_counts = {}
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            if len(source) > 50:
                source = source[:47] + "..."
            source_counts[source] = source_counts.get(source, 0) + 1
        
        if source_counts:
            df_sources = pd.DataFrame(list(source_counts.items()), columns=['Source', 'Document Count'])
            fig = px.bar(df_sources, x='Document Count', y='Source', orientation='h', 
                         title="Documents per Source")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_document_explorer(self):
        """Render document explorer"""
        st.header("🔍 Document Explorer")
        
        if not st.session_state.processed_documents:
            st.info("📋 No documents processed yet")
            return
        
        docs = st.session_state.processed_documents
        
        search_term = st.text_input("🔎 Search within documents:")
        
        if search_term:
            matching_docs = []
            for doc in docs:
                if search_term.lower() in doc.page_content.lower():
                    matching_docs.append(doc)
            
            st.write(f"Found {len(matching_docs)} documents containing '{search_term}'")
            
            for i, doc in enumerate(matching_docs[:10]):
                with st.expander(f"📄 Document {i+1}"):
                    content = doc.page_content
                    if len(content) > 1000:
                        search_pos = content.lower().find(search_term.lower())
                        if search_pos != -1:
                            start = max(0, search_pos - 200)
                            end = min(len(content), search_pos + 300)
                            content = "..." + content[start:end] + "..."
                    
                    st.write("**Content:**")
                    st.write(content)
                    
                    st.write("**Metadata:**")
                    metadata = doc.metadata
                    st.json({k: v for k, v in metadata.items() if k not in ['chunk_id', 'start_index', 'end_index']})
        
        st.subheader("📖 Document Browser")
        
        docs_per_page = 5
        total_pages = (len(docs) - 1) // docs_per_page + 1
        
        if total_pages > 1:
            page = st.selectbox("Select page:", range(1, total_pages + 1))
        else:
            page = 1
        
        start_idx = (page - 1) * docs_per_page
        end_idx = min(start_idx + docs_per_page, len(docs))
        
        for i in range(start_idx, end_idx):
            doc = docs[i]
            source = doc.metadata.get('source', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', '')
            
            display_source = source if len(source) <= 50 else source[:47] + "..."
            
            with st.expander(f"📄 Document {i+1} - {display_source} {f'(Chunk {chunk_id})' if chunk_id else ''}"):
                st.write("**Content:**")
                content = doc.page_content
                st.write(content[:800] + "..." if len(content) > 800 else content)
                
                st.write("**Metadata:**")
                clean_metadata = {k: v for k, v in doc.metadata.items() if k not in ['start_index', 'end_index', 'overlap_start', 'overlap_end']}
                st.json(clean_metadata)
    
    def run_sync(self):
        """Synchronous version of run with async processing support"""
        self.configure_page()
        self.render_header()
        self.render_sidebar()
        
        if hasattr(st.session_state, 'process_button_clicked'):
            if st.session_state.process_button_clicked:
                sources = getattr(st.session_state, 'current_sources', [])
                if sources:
                    success = asyncio.run(self.process_documents_async(sources))
                    if success:
                        st.success("🎉 Documents processed successfully!")
                        st.balloons()
                del st.session_state.process_button_clicked
        
        if hasattr(st.session_state, 'load_button_clicked'):
            if st.session_state.load_button_clicked:
                self.load_existing_data()
                del st.session_state.load_button_clicked
        
        self.render_main_content()

def main():
    """Main function with robust error handling"""
    try:
        app = SmartDocAnalyzer()
        app.run_sync()
        
    except Exception as e:
        st.error(f"""
        ## 🚨 Application Error
        
        **Error:** {str(e)}
        
        **Possible solutions:**
        1. Check that all required files are in place
        2. Verify your Python environment has all dependencies
        3. Ensure the logs directory exists and is writable
        4. Check your OpenAI API key is set correctly
        
        **For development:** Check the console output for detailed error information.
        """)
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()