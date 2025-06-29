"""
SmartDoc Analyzer - Enhanced Configuration Management
Advanced configuration handling with validation and environment support
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API configuration settings"""
    openai_api_key: str
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.2
    max_tokens: int = 4000
    timeout: int = 30

@dataclass
class DocumentProcessingConfig:
    """Document processing configuration"""
    chunk_size: int = 1200
    chunk_overlap: int = 300
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""])
    min_chunk_length: int = 100
    max_chunk_length: int = 2000
    enable_metadata_extraction: bool = True

@dataclass
class VectorDBConfig:
    """Vector database configuration"""
    provider: str = "faiss"
    index_type: str = "flat"
    distance_metric: str = "cosine"
    dimension: int = 1536
    storage_path: str = "./data/vector_stores/"
    index_name: str = "smartdoc_index"

@dataclass
class RetrievalConfig:
    """Retrieval system configuration"""
    search_type: str = "hybrid"
    top_k: int = 5
    similarity_threshold: float = 0.7
    reranking_enabled: bool = True
    reranking_model: str = "cross-encoder"
    diversity_lambda: float = 0.5

@dataclass
class GenerationConfig:
    """Text generation configuration"""
    system_prompt: str
    include_sources: bool = True
    include_confidence: bool = True
    include_metadata: bool = True
    max_response_length: int = 2000

@dataclass
class InterfaceConfig:
    """User interface configuration"""
    title: str = "SmartDoc Analyzer"
    subtitle: str = "Advanced Document Intelligence Research Platform"
    theme: str = "dark"
    layout: str = "wide"
    sidebar_width: int = 350
    max_file_size: int = 50
    supported_formats: List[str] = field(default_factory=lambda: ["txt", "pdf", "docx", "html", "md"])

@dataclass
class DataSourcesConfig:
    """Data sources configuration"""
    web_scraping: Dict[str, Any] = field(default_factory=dict)
    file_upload: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    enable_caching: bool = True
    cache_ttl: int = 3600
    batch_size: int = 32
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit: str = "2GB"

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""
    log_level: str = "INFO"
    log_file: str = "./logs/smartdoc.log"
    enable_metrics: bool = True
    metrics_file: str = "./logs/metrics.json"
    track_usage: bool = True
    track_performance: bool = True

@dataclass
class SecurityConfig:
    """Security configuration"""
    api_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    enable_input_validation: bool = True
    sanitize_uploads: bool = True
    encrypt_storage: bool = False

@dataclass
class ResearchConfig:
    """Research features configuration"""
    enable_experiments: bool = True
    experiment_tracking: bool = True
    a_b_testing: bool = False
    model_comparison: bool = True
    benchmark_mode: bool = False

@dataclass
class AdvancedConfig:
    """Advanced features configuration"""
    multi_language_support: bool = True
    auto_language_detection: bool = True
    summarization_enabled: bool = True
    question_generation: bool = True
    fact_checking: bool = False
    citation_extraction: bool = True

class ConfigManager:
    """Enhanced centralized configuration management system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config_data = {}
        self.metrics = {}
        self._load_config()
        self._validate_config()
        self._setup_logging()
        
    def _load_config(self) -> None:
        """Load configuration from YAML file with environment variable substitution"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                self._create_default_config()
                return
                
            with open(config_file, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Replace environment variables
            content = self._substitute_env_vars(content)
            
            self.config_data = yaml.safe_load(content)
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._create_default_config()
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in config content"""
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        return re.sub(pattern, replace_env_var, content)
    
    def _create_default_config(self) -> None:
        """Create comprehensive default configuration"""
        self.config_data = {
            "api": {
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "model": "gpt-4o-mini",
                    "embedding_model": "text-embedding-3-small",
                    "temperature": 0.2,
                    "max_tokens": 4000,
                    "timeout": 30
                }
            },
            "document_processing": {
                "chunk_size": 1200,
                "chunk_overlap": 300,
                "separators": ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
                "min_chunk_length": 100,
                "max_chunk_length": 2000,
                "enable_metadata_extraction": True
            },
            "vector_db": {
                "provider": "faiss",
                "index_type": "flat",
                "distance_metric": "cosine",
                "dimension": 1536,
                "storage_path": "./data/vector_stores/",
                "index_name": "smartdoc_index"
            },
            "retrieval": {
                "search_type": "hybrid",
                "top_k": 5,
                "similarity_threshold": 0.7,
                "reranking_enabled": True,
                "reranking_model": "cross-encoder",
                "diversity_lambda": 0.5
            },
            "generation": {
                "system_prompt": "You are an advanced document analysis assistant. Provide comprehensive, accurate answers based on the retrieved context. Always cite sources and maintain factual accuracy. Structure your responses clearly and professionally.",
                "response_format": {
                    "include_sources": True,
                    "include_confidence": True,
                    "include_metadata": True,
                    "max_response_length": 2000
                }
            },
            "interface": {
                "title": "SmartDoc Analyzer",
                "subtitle": "Advanced Document Intelligence Research Platform",
                "theme": "dark",
                "layout": "wide",
                "sidebar_width": 350,
                "max_file_size": 50,
                "supported_formats": ["txt", "pdf", "docx", "html", "md"]
            },
            "data_sources": {
                "web_scraping": {
                    "user_agent": "SmartDoc-Analyzer/1.0 (Research Project)",
                    "timeout": 30,
                    "retry_attempts": 3,
                    "retry_delay": 2,
                    "respect_robots_txt": True
                },
                "file_upload": {
                    "max_files": 10,
                    "scan_for_malware": True,
                    "extract_metadata": True
                }
            },
            "performance": {
                "enable_caching": True,
                "cache_ttl": 3600,
                "batch_size": 32,
                "parallel_processing": True,
                "max_workers": 4,
                "memory_limit": "2GB"
            },
            "monitoring": {
                "log_level": "INFO",
                "log_file": "./logs/smartdoc.log",
                "enable_metrics": True,
                "metrics_file": "./logs/metrics.json",
                "track_usage": True,
                "track_performance": True
            },
            "security": {
                "api_rate_limiting": True,
                "max_requests_per_minute": 60,
                "enable_input_validation": True,
                "sanitize_uploads": True,
                "encrypt_storage": False
            },
            "research": {
                "enable_experiments": True,
                "experiment_tracking": True,
                "a_b_testing": False,
                "model_comparison": True,
                "benchmark_mode": False
            },
            "advanced": {
                "multi_language_support": True,
                "auto_language_detection": True,
                "summarization_enabled": True,
                "question_generation": True,
                "fact_checking": False,
                "citation_extraction": True
            }
        }
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration"""
        try:
            monitoring_config = self.get_monitoring_config()
            
            # Set log level
            log_level = getattr(logging, monitoring_config.log_level.upper(), logging.INFO)
            logger.setLevel(log_level)
            
            # Ensure log directory exists
            log_file_path = Path(monitoring_config.log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            logger.warning(f"Error setting up logging: {e}")
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        try:
            # Validate API key
            api_key = self.get_api_config().openai_api_key
            if not api_key or api_key == "your_api_key_here":
                logger.warning("OpenAI API key not configured")
            
            # Validate paths
            storage_path = Path(self.get_vector_db_config().storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Validate numeric ranges
            doc_config = self.get_document_processing_config()
            if doc_config.chunk_overlap >= doc_config.chunk_size:
                logger.warning("Chunk overlap should be less than chunk size")
            
            # Validate performance settings
            perf_config = self.get_performance_config()
            if perf_config.max_workers < 1:
                logger.warning("max_workers should be at least 1")
            
            # Validate security settings
            security_config = self.get_security_config()
            if security_config.max_requests_per_minute < 1:
                logger.warning("max_requests_per_minute should be at least 1")
                
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
    
    # Configuration getter methods
    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        api_data = self.config_data.get("api", {}).get("openai", {})
        return APIConfig(
            openai_api_key=api_data.get("api_key", ""),
            model=api_data.get("model", "gpt-4o-mini"),
            embedding_model=api_data.get("embedding_model", "text-embedding-3-small"),
            temperature=api_data.get("temperature", 0.2),
            max_tokens=api_data.get("max_tokens", 4000),
            timeout=api_data.get("timeout", 30)
        )
    
    def get_document_processing_config(self) -> DocumentProcessingConfig:
        """Get document processing configuration"""
        doc_data = self.config_data.get("document_processing", {})
        return DocumentProcessingConfig(
            chunk_size=doc_data.get("chunk_size", 1200),
            chunk_overlap=doc_data.get("chunk_overlap", 300),
            separators=doc_data.get("separators", ["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]),
            min_chunk_length=doc_data.get("min_chunk_length", 100),
            max_chunk_length=doc_data.get("max_chunk_length", 2000),
            enable_metadata_extraction=doc_data.get("enable_metadata_extraction", True)
        )
    
    def get_vector_db_config(self) -> VectorDBConfig:
        """Get vector database configuration"""
        vdb_data = self.config_data.get("vector_db", {})
        return VectorDBConfig(
            provider=vdb_data.get("provider", "faiss"),
            index_type=vdb_data.get("index_type", "flat"),
            distance_metric=vdb_data.get("distance_metric", "cosine"),
            dimension=vdb_data.get("dimension", 1536),
            storage_path=vdb_data.get("storage_path", "./data/vector_stores/"),
            index_name=vdb_data.get("index_name", "smartdoc_index")
        )
    
    def get_retrieval_config(self) -> RetrievalConfig:
        """Get retrieval configuration"""
        ret_data = self.config_data.get("retrieval", {})
        return RetrievalConfig(
            search_type=ret_data.get("search_type", "hybrid"),
            top_k=ret_data.get("top_k", 5),
            similarity_threshold=ret_data.get("similarity_threshold", 0.7),
            reranking_enabled=ret_data.get("reranking_enabled", True),
            reranking_model=ret_data.get("reranking_model", "cross-encoder"),
            diversity_lambda=ret_data.get("diversity_lambda", 0.5)
        )
    
    def get_generation_config(self) -> GenerationConfig:
        """Get generation configuration"""
        gen_data = self.config_data.get("generation", {})
        response_format = gen_data.get("response_format", {})
        
        default_prompt = """You are an advanced document analysis assistant. Provide comprehensive, accurate answers based on the retrieved context.
        Always cite sources and maintain factual accuracy. Structure your responses clearly and professionally."""
        
        return GenerationConfig(
            system_prompt=gen_data.get("system_prompt", default_prompt),
            include_sources=response_format.get("include_sources", True),
            include_confidence=response_format.get("include_confidence", True),
            include_metadata=response_format.get("include_metadata", True),
            max_response_length=response_format.get("max_response_length", 2000)
        )
    
    def get_interface_config(self) -> InterfaceConfig:
        """Get interface configuration"""
        ui_data = self.config_data.get("interface", {})
        return InterfaceConfig(
            title=ui_data.get("title", "SmartDoc Analyzer"),
            subtitle=ui_data.get("subtitle", "Advanced Document Intelligence Research Platform"),
            theme=ui_data.get("theme", "dark"),
            layout=ui_data.get("layout", "wide"),
            sidebar_width=ui_data.get("sidebar_width", 350),
            max_file_size=ui_data.get("max_file_size", 50),
            supported_formats=ui_data.get("supported_formats", ["txt", "pdf", "docx", "html", "md"])
        )
    
    def get_data_sources_config(self) -> DataSourcesConfig:
        """Get data sources configuration"""
        ds_data = self.config_data.get("data_sources", {})
        return DataSourcesConfig(
            web_scraping=ds_data.get("web_scraping", {}),
            file_upload=ds_data.get("file_upload", {})
        )
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration"""
        perf_data = self.config_data.get("performance", {})
        return PerformanceConfig(
            enable_caching=perf_data.get("enable_caching", True),
            cache_ttl=perf_data.get("cache_ttl", 3600),
            batch_size=perf_data.get("batch_size", 32),
            parallel_processing=perf_data.get("parallel_processing", True),
            max_workers=perf_data.get("max_workers", 4),
            memory_limit=perf_data.get("memory_limit", "2GB")
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        mon_data = self.config_data.get("monitoring", {})
        return MonitoringConfig(
            log_level=mon_data.get("log_level", "INFO"),
            log_file=mon_data.get("log_file", "./logs/smartdoc.log"),
            enable_metrics=mon_data.get("enable_metrics", True),
            metrics_file=mon_data.get("metrics_file", "./logs/metrics.json"),
            track_usage=mon_data.get("track_usage", True),
            track_performance=mon_data.get("track_performance", True)
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        sec_data = self.config_data.get("security", {})
        return SecurityConfig(
            api_rate_limiting=sec_data.get("api_rate_limiting", True),
            max_requests_per_minute=sec_data.get("max_requests_per_minute", 60),
            enable_input_validation=sec_data.get("enable_input_validation", True),
            sanitize_uploads=sec_data.get("sanitize_uploads", True),
            encrypt_storage=sec_data.get("encrypt_storage", False)
        )
    
    def get_research_config(self) -> ResearchConfig:
        """Get research configuration"""
        res_data = self.config_data.get("research", {})
        return ResearchConfig(
            enable_experiments=res_data.get("enable_experiments", True),
            experiment_tracking=res_data.get("experiment_tracking", True),
            a_b_testing=res_data.get("a_b_testing", False),
            model_comparison=res_data.get("model_comparison", True),
            benchmark_mode=res_data.get("benchmark_mode", False)
        )
    
    def get_advanced_config(self) -> AdvancedConfig:
        """Get advanced configuration"""
        adv_data = self.config_data.get("advanced", {})
        return AdvancedConfig(
            multi_language_support=adv_data.get("multi_language_support", True),
            auto_language_detection=adv_data.get("auto_language_detection", True),
            summarization_enabled=adv_data.get("summarization_enabled", True),
            question_generation=adv_data.get("question_generation", True),
            fact_checking=adv_data.get("fact_checking", False),
            citation_extraction=adv_data.get("citation_extraction", True)
        )
    
    # Utility methods
    def get_nested_value(self, path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation"""
        keys = path.split('.')
        value = self.config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def update_config(self, path: str, value: Any) -> None:
        """Update configuration value using dot notation"""
        keys = path.split('.')
        config = self.config_data
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        logger.info(f"Updated config: {path} = {value}")
        
        # Track configuration changes
        self._track_config_change(path, value)
    
    def _track_config_change(self, path: str, value: Any) -> None:
        """Track configuration changes for monitoring"""
        try:
            if self.get_monitoring_config().track_usage:
                change_record = {
                    "timestamp": datetime.now().isoformat(),
                    "path": path,
                    "value": str(value),
                    "type": "config_change"
                }
                
                # Add to metrics
                if "config_changes" not in self.metrics:
                    self.metrics["config_changes"] = []
                
                self.metrics["config_changes"].append(change_record)
                
        except Exception as e:
            logger.warning(f"Error tracking config change: {e}")
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config_data, file, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def validate_api_key(self) -> bool:
        """Validate if API key is properly configured"""
        api_key = self.get_api_config().openai_api_key
        return bool(api_key and api_key != "your_api_key_here" and len(api_key) > 20)
    
    def get_storage_paths(self) -> Dict[str, Path]:
        """Get all storage paths used by the system"""
        base_path = Path("./data")
        return {
            "vector_stores": Path(self.get_vector_db_config().storage_path),
            "uploads": base_path / "uploads",
            "cache": base_path / "cache",
            "logs": Path("./logs"),
            "exports": base_path / "exports",
            "metrics": Path(self.get_monitoring_config().metrics_file).parent
        }
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        paths = self.get_storage_paths()
        for name, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {name} -> {path}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            "api": {
                "model": self.get_api_config().model,
                "embedding_model": self.get_api_config().embedding_model,
                "api_key_configured": self.validate_api_key()
            },
            "processing": {
                "chunk_size": self.get_document_processing_config().chunk_size,
                "chunk_overlap": self.get_document_processing_config().chunk_overlap
            },
            "retrieval": {
                "search_type": self.get_retrieval_config().search_type,
                "top_k": self.get_retrieval_config().top_k
            },
            "performance": {
                "caching_enabled": self.get_performance_config().enable_caching,
                "parallel_processing": self.get_performance_config().parallel_processing
            },
            "features": {
                "experiments_enabled": self.get_research_config().enable_experiments,
                "multi_language": self.get_advanced_config().multi_language_support
            }
        }
    
    def export_metrics(self) -> None:
        """Export metrics to file"""
        try:
            monitoring_config = self.get_monitoring_config()
            if monitoring_config.enable_metrics and self.metrics:
                metrics_path = Path(monitoring_config.metrics_file)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(metrics_path, 'w') as f:
                    json.dump(self.metrics, f, indent=2)
                
                logger.info(f"Metrics exported to {metrics_path}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

# Global configuration instance
_config_manager = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def reload_config(config_path: Optional[str] = None) -> ConfigManager:
    """Reload configuration from file"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager

# Export all configuration classes
__all__ = [
    'ConfigManager', 'get_config_manager', 'reload_config',
    'APIConfig', 'DocumentProcessingConfig', 'VectorDBConfig', 
    'RetrievalConfig', 'GenerationConfig', 'InterfaceConfig',
    'DataSourcesConfig', 'PerformanceConfig', 'MonitoringConfig',
    'SecurityConfig', 'ResearchConfig', 'AdvancedConfig'
]