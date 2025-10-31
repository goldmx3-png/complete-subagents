"""Pydantic schemas for API requests and responses"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request schema"""
    user_id: str = Field(..., description="User ID")
    message: str = Field(..., min_length=1, description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    is_button_click: bool = Field(default=False, description="Whether this is a button click")


class ButtonAction(BaseModel):
    """Button action schema"""
    label: str = Field(..., description="Button label text")
    action: str = Field(..., description="Action to send when clicked")
    icon: Optional[str] = Field(None, description="Button icon (emoji)")
    variant: Optional[str] = Field("primary", description="Button variant: primary, secondary, danger")


class ChatResponse(BaseModel):
    """Chat response schema"""
    conversation_id: str = Field(..., description="Conversation ID")
    message: str = Field(..., description="Assistant response")
    route: str = Field(..., description="Route taken (RAG/MENU/API/SUPPORT)")
    is_menu: bool = Field(default=False, description="Whether response includes a menu")
    buttons: Optional[List[ButtonAction]] = Field(None, description="Interactive buttons to display")
    menu_type: Optional[str] = Field(None, description="Type of menu: main, accounts, payments, documents, support")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component statuses")


class UploadResponse(BaseModel):
    """Document upload response"""
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Status message")


class DocumentInfo(BaseModel):
    """Document information"""
    doc_id: str = Field(..., description="Document ID")
    file_name: Optional[str] = Field(None, description="Filename")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    chunk_count: int = Field(default=0, description="Number of chunks")


class ConversationInfo(BaseModel):
    """Conversation information"""
    conversation_id: str = Field(..., description="Conversation ID")
    title: str = Field(..., description="Conversation title")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    message_count: Optional[int] = Field(None, description="Number of messages")


class MessageInfo(BaseModel):
    """Message information"""
    message_id: str = Field(..., description="Message ID")
    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    created_at: str = Field(..., description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default={}, description="Message metadata")
