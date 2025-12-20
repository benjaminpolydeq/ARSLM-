"""
ARSLM API - FastAPI Application

REST API for ARSLM model inference and management.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uvicorn
import torch
from datetime import datetime
import os
from pathlib import Path

# Import local modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.arslm.core.model import ARSLMModel, ARSLMConfig
from src.arslm.utils.tokenizer import SimpleTokenizer

# Initialize FastAPI app
app = FastAPI(
    title="ARSLM API",
    description="Adaptive Reasoning Semantic Language Model API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
MODEL = None
TOKENIZER = None
CONVERSATIONS = {}  # In-memory conversation storage


# Pydantic schemas
class ChatRequest(BaseModel):
    """Chat request schema."""
    message: str = Field(..., description="User message")
    session_id: str = Field(..., description="Session identifier")
    max_length: int = Field(100, ge=1, le=500, description="Maximum response length")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling")


class ChatResponse(BaseModel):
    """Chat response schema."""
    response: str = Field(..., description="Generated response")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(..., description="Response timestamp")
    confidence: float = Field(..., description="Response confidence score")
    tokens_generated: int = Field(..., description="Number of tokens generated")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    timestamp: str


class HistoryResponse(BaseModel):
    """Conversation history response."""
    session_id: str
    messages: List[Dict[str, str]]
    message_count: int


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global MODEL, TOKENIZER
    
    print("🚀 Starting ARSLM API...")
    
    # Initialize tokenizer
    TOKENIZER = SimpleTokenizer(vocab_size=50000)
    print("✅ Tokenizer initialized")
    
    # Load or create model
    model_path = os.getenv("MODEL_PATH", "./models/arslm_base.pt")
    
    if os.path.exists(model_path):
        print(f"📦 Loading model from {model_path}")
        try:
            MODEL = ARSLMModel.from_pretrained(model_path)
            MODEL.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("🔧 Creating new model...")
            config = ARSLMConfig()
            MODEL = ARSLMModel(config)
            MODEL.eval()
            print("✅ New model created")
    else:
        print("🔧 Creating new model (no checkpoint found)")
        config = ARSLMConfig()
        MODEL = ARSLMModel(config)
        MODEL.eval()
        print("✅ Model ready")


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=MODEL is not None,
        timestamp=datetime.now().isoformat()
    )


# Chat endpoint
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Generate chat response.
    
    Args:
        request: Chat request with message and parameters
        
    Returns:
        Generated response
    """
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Get or create conversation history
        if request.session_id not in CONVERSATIONS:
            CONVERSATIONS[request.session_id] = []
        
        # Add user message to history
        CONVERSATIONS[request.session_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Prepare context (last 5 messages)
        context = CONVERSATIONS[request.session_id][-5:]
        context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        
        # Tokenize input
        input_ids = TOKENIZER.encode(context_text)
        input_tensor = torch.tensor([input_ids])
        
        # Generate response
        with torch.no_grad():
            output_ids = MODEL.generate(
                input_tensor,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            )
        
        # Decode response
        response_text = TOKENIZER.decode(output_ids[0].tolist())
        
        # Extract only the new generated part
        if "assistant:" in response_text.lower():
            response_text = response_text.split("assistant:")[-1].strip()
        
        # Clean up response
        response_text = response_text.replace(context_text, "").strip()
        if not response_text:
            response_text = "I understand. How can I help you further?"
        
        # Calculate confidence (simplified)
        confidence = 0.85  # Placeholder
        
        # Add assistant response to history
        CONVERSATIONS[request.session_id].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            tokens_generated=len(output_ids[0]) - len(input_ids)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {str(e)}"
        )


# Get conversation history
@app.get("/api/v1/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """
    Get conversation history for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Conversation history
    """
    if session_id not in CONVERSATIONS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    return HistoryResponse(
        session_id=session_id,
        messages=CONVERSATIONS[session_id],
        message_count=len(CONVERSATIONS[session_id])
    )


# Clear conversation history
@app.delete("/api/v1/history/{session_id}")
async def clear_history(session_id: str):
    """
    Clear conversation history for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
    """
    if session_id in CONVERSATIONS:
        del CONVERSATIONS[session_id]
        return {"message": f"History for session {session_id} cleared successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )


# List all sessions
@app.get("/api/v1/sessions")
async def list_sessions():
    """
    List all active sessions.
    
    Returns:
        List of session IDs
    """
    return {
        "sessions": list(CONVERSATIONS.keys()),
        "total_sessions": len(CONVERSATIONS)
    }


# Model info endpoint
@app.get("/api/v1/model/info")
async def model_info():
    """
    Get model information.
    
    Returns:
        Model configuration and stats
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in MODEL.parameters())
    trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    
    return {
        "model_type": "ARSLM",
        "version": "1.0.0",
        "config": {
            "vocab_size": MODEL.config.vocab_size,
            "d_model": MODEL.config.d_model,
            "n_heads": MODEL.config.n_heads,
            "n_layers": MODEL.config.n_layers,
            "max_seq_length": MODEL.config.max_seq_length
        },
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "size_mb": total_params * 4 / (1024 ** 2)  # Assuming float32
        }
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Run server
def run():
    """Run the API server."""
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        workers=int(os.getenv("API_WORKERS", 1))
    )


if __name__ == "__main__":
    run()
