"""FastAPI routes for subagent RAG system"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from src.api.schemas import (
    ChatRequest, ChatResponse, HealthResponse,
    UploadResponse, DocumentInfo, ConversationInfo, MessageInfo
)
from src.agents.orchestrator import OrchestratorAgent
from src.memory.conversation_store import get_conversation_store
from src.document_processing.uploader import DocumentUploader
from src.config import settings
from src.utils.logger import get_logger
from src.utils.correlation import set_correlation_id
import uuid
import shutil
import json
from pathlib import Path
from typing import List

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Subagents RAG API",
    description="Modular RAG system with domain-based subagents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (singletons)
orchestrator = OrchestratorAgent()
conversation_store = get_conversation_store()
uploader = DocumentUploader()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Subagents RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "api": "ok",
            "orchestrator": "ok",
            "llm": "ok",
            "vectorstore": "ok"
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint

    Processes user queries through the orchestrator agent
    """
    # Set correlation ID for request tracking
    correlation_id = set_correlation_id()
    logger.info(f"Chat request: user_id={request.user_id}, message_len={len(request.message)}")

    try:
        # Get or create conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = await conversation_store.create_conversation(request.user_id)

        # Get conversation history
        history = await conversation_store.get_conversation_history(conversation_id, limit=5)

        # Run orchestrator
        result = await orchestrator.run(
            query=request.message,
            user_id=request.user_id,
            conversation_id=conversation_id,
            conversation_history=history,
            is_button_click=request.is_button_click
        )

        # Save messages to conversation
        await conversation_store.save_message(conversation_id, "user", request.message, request.user_id)
        await conversation_store.save_message(conversation_id, "assistant", result["final_response"], request.user_id)

        # Build response
        from src.api.schemas import ButtonAction

        # Get menu data from nested structure
        menu_data = result.get("menu", {})

        # Convert menu buttons to ButtonAction objects
        buttons = None
        if menu_data.get("menu_buttons"):
            buttons = [ButtonAction(**btn) for btn in menu_data["menu_buttons"]]

        response = ChatResponse(
            conversation_id=conversation_id,
            message=result["final_response"],
            route=result.get("route", "unknown"),
            is_menu=menu_data.get("is_menu", False),
            buttons=buttons,
            menu_type=menu_data.get("menu_type"),
            metadata={
                "route_reasoning": result.get("route_reasoning", ""),
                "correlation_id": correlation_id,
                "is_menu": menu_data.get("is_menu", False),
                "buttons": [btn.dict() for btn in buttons] if buttons else None,
                "menu_type": menu_data.get("menu_type")
            }
        )

        logger.info(f"Chat response: route={response.route}, response_len={len(response.message)}")
        return response

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - returns Server-Sent Events (SSE)

    Args:
        request: Chat request with user message

    Returns:
        StreamingResponse with SSE events
    """
    correlation_id = set_correlation_id()
    logger.info(f"POST /chat/stream received from user_id={request.user_id}")

    async def event_generator():
        try:
            # Get or create conversation
            conversation_id = request.conversation_id
            if not conversation_id:
                conversation_id = await conversation_store.create_conversation(request.user_id)

            # Send conversation_id first
            yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conversation_id})}\n\n"

            # Get conversation history
            history = await conversation_store.get_conversation_history(conversation_id, limit=5)

            # Save user message
            await conversation_store.save_message(conversation_id, "user", request.message, request.user_id)

            # Run orchestrator
            result = await orchestrator.run(
                query=request.message,
                user_id=request.user_id,
                conversation_id=conversation_id,
                conversation_history=history,
                is_button_click=request.is_button_click
            )

            # Stream the response in chunks
            response_text = result["final_response"]
            chunk_size = 50
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

            # Get menu data from nested structure
            menu_data = result.get("menu", {})

            # Build metadata with menu information
            metadata = {
                'type': 'done',
                'route': result.get("route", "unknown"),
                'correlation_id': correlation_id,
                'is_menu': menu_data.get("is_menu", False),
                'menu_type': menu_data.get("menu_type")
            }

            # Add buttons if present
            if menu_data.get("menu_buttons"):
                metadata['buttons'] = menu_data["menu_buttons"]

            yield f"data: {json.dumps(metadata)}\n\n"

            # Save assistant message
            await conversation_store.save_message(conversation_id, "assistant", response_text, request.user_id)

            # Send completion event
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("default"),
    extract_tables: bool = Form(True)
):
    """
    Upload and process document

    Args:
        file: Uploaded file
        user_id: User ID
        extract_tables: Whether to extract tables

    Returns:
        Upload processing result
    """
    correlation_id = set_correlation_id()
    logger.info(f"POST /upload received: file={file.filename}, user_id={user_id}")

    temp_file_path = None

    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        temp_file_path = temp_dir / file.filename
        logger.debug(f"Saving uploaded file to {temp_file_path}")

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved, starting document processing")

        # Process document
        result = await uploader.upload_document(
            file_path=str(temp_file_path),
            user_id=user_id,
            extract_tables=extract_tables
        )

        # Clean up temp file
        temp_file_path.unlink()
        logger.debug("Temporary file cleaned up")

        logger.info(f"POST /upload completed successfully: doc_id={result.get('document_id')}")

        return UploadResponse(**result)

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)

        # Clean up temp file on error
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{user_id}", response_model=List[DocumentInfo])
async def list_documents(user_id: str):
    """
    List all documents for a user

    Args:
        user_id: User ID

    Returns:
        List of document information
    """
    try:
        # Get document IDs from upload directory
        upload_dir = Path("uploads")
        documents = []

        if upload_dir.exists():
            for file_path in upload_dir.glob("*"):
                if file_path.is_file():
                    # Extract doc_id from filename (format: {doc_id}_{original_name})
                    parts = file_path.name.split("_", 1)
                    if len(parts) == 2:
                        doc_id = parts[0]

                        # Get document info
                        try:
                            doc_info = await uploader.get_document_info(doc_id, user_id)
                            documents.append(DocumentInfo(
                                doc_id=doc_id,
                                file_name=parts[1],
                                chunk_count=doc_info.get("chunks_count", 0),
                                file_size=file_path.stat().st_size
                            ))
                        except Exception as e:
                            logger.warning(f"Error getting info for doc {doc_id}: {e}")

        logger.info(f"GET /documents/{user_id} completed: {len(documents)} documents found")
        return documents

    except Exception as e:
        logger.error(f"Error listing documents for user_id={user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user_id: str):
    """
    Delete a document

    Args:
        doc_id: Document ID
        user_id: User ID

    Returns:
        Deletion result
    """
    logger.info(f"DELETE /documents/{doc_id} received: user_id={user_id}")

    try:
        result = await uploader.delete_document(doc_id, user_id)
        logger.info(f"DELETE /documents/{doc_id} completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{user_id}", response_model=List[ConversationInfo])
async def list_conversations(user_id: str, limit: int = 50):
    """
    List all conversations for a user

    Args:
        user_id: User ID
        limit: Maximum conversations to return

    Returns:
        List of conversations
    """
    try:
        conversations = await conversation_store.get_user_conversations(user_id, limit)
        logger.info(f"GET /conversations/{user_id} completed: {len(conversations)} conversations found")

        # Convert to ConversationInfo format
        return [
            ConversationInfo(
                conversation_id=conv["conversation_id"],
                title=conv["title"],
                created_at=conv["created_at"],
                updated_at=conv["updated_at"]
            )
            for conv in conversations
        ]

    except Exception as e:
        logger.error(f"Error listing conversations for user_id={user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageInfo])
async def get_conversation_messages(conversation_id: str):
    """
    Get all messages in a conversation

    Args:
        conversation_id: Conversation ID

    Returns:
        List of messages
    """
    try:
        messages = await conversation_store.get_conversation_history_full(conversation_id)
        logger.info(f"GET /conversations/{conversation_id}/messages completed: {len(messages)} messages found")

        # Convert to MessageInfo format
        return [
            MessageInfo(
                message_id=msg["message_id"],
                role=msg["role"],
                content=msg["content"],
                created_at=msg["created_at"],
                metadata=msg.get("metadata", {})
            )
            for msg in messages
        ]

    except Exception as e:
        logger.error(f"Error getting messages for conversation_id={conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation

    Args:
        conversation_id: Conversation ID

    Returns:
        Deletion result
    """
    logger.info(f"DELETE /conversations/{conversation_id} received")

    try:
        await conversation_store.delete_conversation(conversation_id)
        logger.info(f"DELETE /conversations/{conversation_id} completed successfully")
        return {"status": "deleted", "conversation_id": conversation_id}

    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
