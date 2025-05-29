from fastapi import FastAPI, Body, HTTPException, Query
from pydantic import BaseModel, Field
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Retriever Agent", version="2.0.0")

# Configuration
EMBEDDING_DIM = 384
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index.bin"
DOCS_PATH = "documents.pkl"
METADATA_PATH = "metadata.pkl"
MAX_MEMORY_MB = 512  # Maximum memory usage in MB
MAX_DOCUMENTS = 10000  # Maximum number of documents

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class IndexRequest(BaseModel):
    documents: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None
    batch_size: int = Field(default=32, ge=1, le=1000)

class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=3, ge=1, le=50)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    filter_metadata: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime

class IndexResponse(BaseModel):
    status: str
    indexed_count: int
    total_documents: int
    duplicates_skipped: int
    memory_usage_mb: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int
    search_time_ms: float

class RetrieverAgent:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.documents: List[Document] = []
        self.doc_hashes: set = set()  # For duplicate detection
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.RLock()  # For thread-safe operations
        
        # Load existing data
        self._load_index()
        logger.info(f"Initialized with {len(self.documents)} documents")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    def _check_memory_limits(self):
        """Check if memory usage is within limits"""
        current_memory = self._get_memory_usage()
        if current_memory > MAX_MEMORY_MB:
            logger.warning(f"Memory usage ({current_memory:.1f}MB) exceeds limit ({MAX_MEMORY_MB}MB)")
            gc.collect()  # Force garbage collection

    def _generate_hash(self, content: str) -> str:
        """Generate hash for duplicate detection"""
        return hashlib.md5(content.encode()).hexdigest()

    def _load_index(self):
        """Load existing index and documents with proper error handling"""
        with self._lock:
            try:
                if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
                    # Load FAISS index
                    self.index = faiss.read_index(INDEX_PATH)
                    
                    # Load documents
                    with open(DOCS_PATH, 'rb') as f:
                        self.documents = pickle.load(f)
                    
                    # Rebuild hash set
                    self.doc_hashes = {self._generate_hash(doc.content) for doc in self.documents}
                    
                    # Validate index consistency
                    if self.index.ntotal != len(self.documents):
                        logger.warning("Index-document count mismatch. Rebuilding index...")
                        self._rebuild_index()
                    
                    logger.info(f"Loaded {len(self.documents)} documents from disk")
                else:
                    logger.info("No existing index found. Starting fresh.")
                    
            except Exception as e:
                logger.error(f"Could not load existing index: {e}")
                self._reset_index()

    def _rebuild_index(self):
        """Rebuild FAISS index from existing documents"""
        try:
            if not self.documents:
                self._reset_index()
                return
                
            contents = [doc.content for doc in self.documents]
            embeddings = self.model.encode(contents)
            
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.index.add(np.array(embeddings))
            
            logger.info(f"Rebuilt index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            self._reset_index()

    def _save_index(self):
        """Save index and documents to disk with proper error handling"""
        with self._lock:
            try:
                # Create backup of existing files
                backup_files = []
                for file_path in [INDEX_PATH, DOCS_PATH]:
                    if os.path.exists(file_path):
                        backup_path = f"{file_path}.backup"
                        os.rename(file_path, backup_path)
                        backup_files.append((file_path, backup_path))
                
                # Save new files
                faiss.write_index(self.index, INDEX_PATH)
                
                with open(DOCS_PATH, 'wb') as f:
                    pickle.dump(self.documents, f)
                
                # Remove backups on success
                for _, backup_path in backup_files:
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                        
                logger.info("Index saved successfully")
                
            except Exception as e:
                logger.error(f"Failed to save index: {e}")
                
                # Restore backups on failure
                for original_path, backup_path in backup_files:
                    if os.path.exists(backup_path):
                        os.rename(backup_path, original_path)

    def _reset_index(self):
        """Reset index to empty state"""
        with self._lock:
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.documents = []
            self.doc_hashes = set()

    async def index_documents(self, request: IndexRequest) -> IndexResponse:
        """Index documents with duplicate detection and batch processing"""
        start_time = datetime.now()
        
        # Check document limits
        if len(self.documents) + len(request.documents) > MAX_DOCUMENTS:
            raise HTTPException(
                status_code=413, 
                detail=f"Document limit exceeded. Maximum: {MAX_DOCUMENTS}"
            )
        
        new_docs = []
        duplicates_skipped = 0
        
        # Prepare documents and detect duplicates
        with self._lock:
            for i, content in enumerate(request.documents):
                content_hash = self._generate_hash(content)
                
                if content_hash in self.doc_hashes:
                    duplicates_skipped += 1
                    continue
                
                metadata = None
                if request.metadata and i < len(request.metadata):
                    metadata = request.metadata[i]
                
                doc = Document(content=content, metadata=metadata)
                new_docs.append(doc)
                self.doc_hashes.add(content_hash)

        if not new_docs:
            return IndexResponse(
                status="completed",
                indexed_count=0,
                total_documents=len(self.documents),
                duplicates_skipped=duplicates_skipped,
                memory_usage_mb=self._get_memory_usage()
            )

        # Generate embeddings in batches
        contents = [doc.content for doc in new_docs]
        
        try:
            # Run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor, 
                self.model.encode, 
                contents
            )
            
            # Add to FAISS index with lock
            with self._lock:
                self.index.add(np.array(embeddings))
                self.documents.extend(new_docs)
            
            # Save to disk
            self._save_index()
            
            # Check memory usage
            self._check_memory_limits()
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Indexed {len(new_docs)} documents in {duration:.2f}s")
            
            return IndexResponse(
                status="completed",
                indexed_count=len(new_docs),
                total_documents=len(self.documents),
                duplicates_skipped=duplicates_skipped,
                memory_usage_mb=self._get_memory_usage()
            )
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    async def search_documents(self, request: SearchRequest) -> SearchResponse:
        """Search documents with filtering and scoring"""
        start_time = datetime.now()
        
        if len(self.documents) == 0:
            return SearchResponse(
                query=request.query,
                results=[],
                total_found=0,
                search_time_ms=0.0
            )
        
        try:
            # Generate query embedding
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                self.executor,
                self.model.encode,
                [request.query]
            )
            
            # Search FAISS index with lock
            with self._lock:
                distances, indices = self.index.search(
                    np.array(query_embedding), 
                    min(request.k * 2, len(self.documents))  # Get more for filtering
                )
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx >= len(self.documents) or idx < 0:  # Safety check
                    continue
                    
                doc = self.documents[idx]
                score = float(distance)
                
                # Apply threshold filter
                if request.threshold is not None and score > request.threshold:
                    continue
                
                # Apply metadata filter
                if request.filter_metadata and doc.metadata:
                    if not self._matches_filter(doc.metadata, request.filter_metadata):
                        continue
                
                results.append(SearchResult(
                    id=doc.id,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata,
                    timestamp=doc.timestamp
                ))
                
                if len(results) >= request.k:
                    break
            
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return SearchResponse(
                query=request.query,
                results=results,
                total_found=len(results),
                search_time_ms=search_time
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    def _matches_filter(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches filter criteria"""
        for key, value in filters.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "embedding_dimension": EMBEDDING_DIM,
            "model_name": MODEL_NAME,
            "index_type": "FAISS L2",
            "memory_usage_mb": self._get_memory_usage(),
            "max_memory_mb": MAX_MEMORY_MB,
            "max_documents": MAX_DOCUMENTS
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.executor.shutdown(wait=True)
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Global agent instance
agent = RetrieverAgent()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    agent.cleanup()

@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """Index documents with metadata and duplicate detection"""
    return await agent.index_documents(request)

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents with advanced filtering"""
    return await agent.search_documents(request)

@app.get("/search", response_model=SearchResponse)
async def search_simple(
    query: str = Query(..., description="Search query"),
    k: int = Query(3, ge=1, le=50, description="Number of results"),
    threshold: Optional[float] = Query(None, ge=0.0, le=2.0, description="Distance threshold")
):
    """Simple GET endpoint for search"""
    request = SearchRequest(query=query, k=k, threshold=threshold)
    return await agent.search_documents(request)

@app.get("/stats")
async def get_statistics():
    """Get index statistics"""
    return agent.get_stats()

@app.delete("/index")
async def clear_index():
    """Clear all documents from index"""
    agent._reset_index()
    agent._save_index()
    return {"status": "cleared", "total_documents": 0}

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get specific document by ID"""
    with agent._lock:
        for doc in agent.documents:
            if doc.id == doc_id:
                return doc
    raise HTTPException(status_code=404, detail="Document not found")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete specific document (requires index rebuild)"""
    with agent._lock:
        doc_index = None
        for i, doc in enumerate(agent.documents):
            if doc.id == doc_id:
                doc_index = i
                break
        
        if doc_index is None:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove document and rebuild index
        removed_doc = agent.documents.pop(doc_index)
        agent.doc_hashes.discard(agent._generate_hash(removed_doc.content))
        
        # Rebuild index
        if agent.documents:
            contents = [doc.content for doc in agent.documents]
            embeddings = agent.model.encode(contents)
            agent.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            agent.index.add(np.array(embeddings))
        else:
            agent._reset_index()
        
        agent._save_index()
    
    return {"status": "deleted", "document_id": doc_id}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": agent.model is not None,
        "total_documents": len(agent.documents),
        "memory_usage_mb": agent._get_memory_usage()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)