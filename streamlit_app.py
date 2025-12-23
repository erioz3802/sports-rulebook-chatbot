"""
Sports Rulebook Q&A - Streamlit App
A RAG-based chatbot for answering questions about sports rules.
"""

import streamlit as st
import sqlite3
import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import math

# Config file for saving settings
CONFIG_FILE = "config.json"

def load_config() -> Dict:
    """Load configuration from file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(config: Dict):
    """Save configuration to file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)
    except:
        pass  # May fail on cloud deployment, that's OK

def get_api_key() -> Optional[str]:
    """Get API key from Streamlit secrets (cloud) or config file (local)."""
    # First try Streamlit secrets (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
            return st.secrets['ANTHROPIC_API_KEY']
    except:
        pass
    
    # Fall back to config file (for local use)
    config = load_config()
    return config.get("api_key")

def save_api_key(api_key: str):
    """Save API key to config."""
    config = load_config()
    config["api_key"] = api_key
    save_config(config)

# Optional imports with fallbacks
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Configuration
DB_PATH = "rules_qa.db"
DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

CATEGORIES = [
    {"id": "baseball", "name": "Baseball", "icon": "⚾"},
    {"id": "basketball", "name": "Basketball", "icon": "🏀"},
    {"id": "football", "name": "Football", "icon": "🏈"},
    {"id": "soccer", "name": "Soccer", "icon": "⚽"},
    {"id": "hockey", "name": "Hockey", "icon": "🏒"},
    {"id": "volleyball", "name": "Volleyball", "icon": "🏐"},
    {"id": "tennis", "name": "Tennis", "icon": "🎾"},
    {"id": "golf", "name": "Golf", "icon": "⛳"},
    {"id": "swimming", "name": "Swimming", "icon": "🏊"},
    {"id": "track", "name": "Track & Field", "icon": "🏃"},
    {"id": "softball", "name": "Softball", "icon": "🥎"},
    {"id": "other", "name": "Other", "icon": "🏆"},
]

LEVELS = [
    {"id": "little_league", "name": "Little League"},
    {"id": "youth", "name": "Youth"},
    {"id": "middle_school", "name": "Middle School"},
    {"id": "jv", "name": "Junior Varsity"},
    {"id": "varsity", "name": "Varsity"},
    {"id": "high_school", "name": "High School"},
    {"id": "club", "name": "Club"},
    {"id": "college", "name": "College"},
    {"id": "semi_pro", "name": "Semi-Pro"},
    {"id": "professional", "name": "Professional"},
    {"id": "international", "name": "International"},
]

# Database functions
def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            category TEXT NOT NULL,
            level TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            file_size INTEGER,
            chunk_count INTEGER DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            rule_references TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            category TEXT,
            level TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_level ON documents(level)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
    
    conn.commit()
    conn.close()

def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)

# Conversation management functions
def create_conversation(title: str = "New Conversation", category: str = None, level: str = None) -> str:
    """Create a new conversation and return its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    conv_id = hashlib.md5(f"{title}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    now = datetime.now().isoformat()
    
    cursor.execute("""
        INSERT INTO conversations (id, title, created_at, updated_at, category, level)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (conv_id, title, now, now, category, level))
    
    conn.commit()
    conn.close()
    return conv_id

def get_conversations(limit: int = 50) -> List[Dict]:
    """Get all conversations, most recent first."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, title, created_at, updated_at, category, level
        FROM conversations
        ORDER BY updated_at DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "category": row[4],
            "level": row[5]
        }
        for row in rows
    ]

def update_conversation_title(conv_id: str, title: str):
    """Update conversation title."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE conversations 
        SET title = ?, updated_at = ?
        WHERE id = ?
    """, (title, datetime.now().isoformat(), conv_id))
    
    conn.commit()
    conn.close()

def delete_conversation(conv_id: str):
    """Delete a conversation and its messages."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    
    conn.commit()
    conn.close()

def save_message(conv_id: str, role: str, content: str, metadata: Dict = None):
    """Save a message to a conversation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO messages (conversation_id, role, content, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (conv_id, role, content, json.dumps(metadata) if metadata else None, datetime.now().isoformat()))
    
    # Update conversation's updated_at
    cursor.execute("""
        UPDATE conversations SET updated_at = ? WHERE id = ?
    """, (datetime.now().isoformat(), conv_id))
    
    conn.commit()
    conn.close()

def get_messages(conv_id: str) -> List[Dict]:
    """Get all messages for a conversation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT role, content, metadata, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
    """, (conv_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        msg = {
            "role": row[0],
            "content": row[1],
            "created_at": row[3]
        }
        if row[2]:
            try:
                msg["metadata"] = json.loads(row[2])
            except:
                pass
        messages.append(msg)
    
    return messages

def generate_title_from_question(question: str) -> str:
    """Generate a short title from the first question."""
    # Take first 50 chars and clean up
    title = question[:50].strip()
    if len(question) > 50:
        title += "..."
    return title

# Document processing functions
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    if not PDF_SUPPORT:
        raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")
    
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from Word document."""
    if not DOCX_SUPPORT:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    
    doc = DocxDocument(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file_path: str) -> str:
    """Extract text from various file types."""
    ext = Path(file_path).suffix.lower()
    
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def extract_rule_references(text: str) -> List[str]:
    """Extract rule references from text."""
    patterns = [
        r'Rule\s+(\d+(?:\.\d+)*(?:\([a-zA-Z]\))?)',
        r'Section\s+(\d+(?:\.\d+)*)',
        r'Article\s+(\d+(?:\.\d+)*)',
        r'(\d+\.\d+(?:\.\d+)?(?:\([a-zA-Z]\))?)',
    ]
    
    references = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            references.add(match)
    
    return list(references)[:10]

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                chunk_text = chunk_text[:break_point + 1]
                end = start + break_point + 1
        
        rule_refs = extract_rule_references(chunk_text)
        
        chunks.append({
            "content": chunk_text.strip(),
            "chunk_index": chunk_index,
            "rule_references": rule_refs
        })
        
        chunk_index += 1
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

def process_document(file_path: str, filename: str, category: str, level: str) -> str:
    """Process and store a document."""
    # Extract text
    text = extract_text(file_path)
    
    if not text.strip():
        raise ValueError("No text content found in document")
    
    # Generate document ID
    doc_id = hashlib.md5(f"{filename}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    # Chunk the text
    chunks = chunk_text(text)
    
    # Store in database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Insert document
        cursor.execute("""
            INSERT INTO documents (id, filename, category, level, upload_date, file_size, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, filename, category, level, datetime.now().isoformat(), 
              os.path.getsize(file_path), len(chunks)))
        
        # Insert chunks
        for chunk in chunks:
            chunk_id = hashlib.md5(f"{doc_id}_{chunk['chunk_index']}".encode()).hexdigest()[:12]
            cursor.execute("""
                INSERT INTO chunks (id, document_id, content, chunk_index, rule_references)
                VALUES (?, ?, ?, ?, ?)
            """, (chunk_id, doc_id, chunk["content"], chunk["chunk_index"], 
                  json.dumps(chunk["rule_references"])))
        
        conn.commit()
        return doc_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# Search functions (TF-IDF based)
def tokenize(text: str) -> List[str]:
    """Simple tokenization."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) > 2]

def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute term frequency."""
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    # Normalize
    max_freq = max(tf.values()) if tf else 1
    return {k: v / max_freq for k, v in tf.items()}

def search_chunks(query: str, category: Optional[str] = None, level: Optional[str] = None, document_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """Search for relevant chunks using improved TF-IDF-like scoring."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build query
    sql = """
        SELECT c.id, c.content, c.rule_references, d.filename, d.category, d.level
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE 1=1
    """
    params = []
    
    # Filter by specific document if provided
    if document_id:
        sql += " AND d.id = ?"
        params.append(document_id)
    else:
        # Only apply category/level filters if no specific document selected
        if category:
            sql += " AND d.category = ?"
            params.append(category)
        
        if level:
            sql += " AND d.level = ?"
            params.append(level)
    
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
    
    # Tokenize query and expand with related terms
    query_tokens = tokenize(query)
    query_lower = query.lower()
    
    # Calculate IDF for query terms across all chunks
    doc_freq = {}
    for row in rows:
        chunk_tokens = set(tokenize(row[1]))
        for token in chunk_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    
    total_docs = len(rows)
    
    # Score each chunk
    scored_chunks = []
    for row in rows:
        chunk_id, content, rule_refs, filename, cat, lvl = row
        content_lower = content.lower()
        chunk_tokens = tokenize(content)
        chunk_tf = compute_tf(chunk_tokens)
        
        # TF-IDF scoring
        score = 0
        for token in query_tokens:
            if token in chunk_tf:
                tf = chunk_tf[token]
                idf = math.log(total_docs / (1 + doc_freq.get(token, 0)))
                score += tf * idf
        
        # Boost for exact phrase matches
        if query_lower in content_lower:
            score += 2.0
        
        # Boost for rule references matching query
        try:
            refs = json.loads(rule_refs) if rule_refs else []
        except:
            refs = []
        
        for token in query_tokens:
            for ref in refs:
                if token in ref.lower():
                    score += 0.5
        
        # Boost for multiple query term matches
        matching_terms = sum(1 for t in query_tokens if t in chunk_tf)
        if matching_terms >= 2:
            score += matching_terms * 0.3
        
        if score > 0:
            scored_chunks.append({
                "id": chunk_id,
                "content": content,
                "rule_references": refs,
                "filename": filename,
                "category": cat,
                "level": lvl,
                "score": score
            })
    
    # Sort by score and return top results
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:limit]

# Q&A functions
def generate_answer_with_claude(question: str, context: str, api_key: str, conversation_history: List[Dict] = None) -> Dict:
    """Generate answer using Claude API with conversation history."""
    if not HTTPX_AVAILABLE:
        return generate_fallback_answer(question, context)
    
    try:
        # Build messages with conversation history
        messages = []
        
        # Add conversation history (last 10 exchanges to stay within limits)
        if conversation_history:
            recent_history = conversation_history[-20:]  # Last 10 Q&A pairs (20 messages)
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    # Only include the answer text, not metadata
                    messages.append({"role": "assistant", "content": msg["content"]})
        
        # Add current question with context
        current_prompt = f"""You are an expert sports rules analyst and officiating consultant. Your job is to provide comprehensive, well-analyzed answers to rules questions based on official rulebook content.

RULEBOOK EXCERPTS:
{context}

QUESTION: {question}

Please provide a thorough analysis following this structure:

1. **Direct Answer**: Start with a clear, direct answer to the question.

2. **Rule Analysis**: Cite the specific rule(s) that apply, including rule numbers/sections when available. Explain what the rule states and how it applies to this situation.

3. **Key Considerations**: Identify any important factors, exceptions, or edge cases that could affect the ruling. Consider different scenarios or interpretations.

4. **Practical Application**: Explain how this would be applied in a real game situation. Include any signals, procedures, or common mistakes to avoid.

5. **Related Rules**: Mention any related rules that might also be relevant or that are commonly confused with this situation.

If this is a follow-up question, use the conversation context to provide a relevant and connected answer.

If the rulebook excerpts don't contain enough information to fully answer the question, clearly state what information is missing and provide what guidance you can based on general rules knowledge.

Be thorough but clear. Use specific rule references whenever possible."""
        
        messages.append({"role": "user", "content": current_prompt})
        
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2048,
                    "system": "You are a helpful sports rules expert assistant. You remember the conversation history and can answer follow-up questions in context. When users ask clarifying questions or say things like 'what about...' or 'and if...', relate your answer to the previous discussion.",
                    "messages": messages
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["content"][0]["text"]
                return {
                    "answer": answer,
                    "confidence": 0.9,
                    "source": "claude"
                }
            else:
                return generate_fallback_answer(question, context)
    except Exception as e:
        return generate_fallback_answer(question, context)

def generate_fallback_answer(question: str, context: str) -> Dict:
    """Generate a comprehensive answer without API."""
    if not context.strip():
        return {
            "answer": "I couldn't find any relevant information in the uploaded rulebooks. Please make sure you've uploaded rulebooks for the sport and level you're asking about.",
            "confidence": 0.0,
            "source": "fallback"
        }
    
    # Extract sentences containing question keywords
    question_tokens = set(tokenize(question))
    sentences = re.split(r'[.!?]+', context)
    relevant_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        sentence_tokens = set(tokenize(sentence))
        overlap = len(question_tokens & sentence_tokens)
        if overlap >= 1:
            relevant_sentences.append((sentence, overlap))
    
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in relevant_sentences[:8]]
    
    # Extract rule references from context
    rule_refs = extract_rule_references(context)
    
    if top_sentences:
        answer = "## 📋 Analysis Based on Rulebook Content\n\n"
        answer += "### Relevant Rule Excerpts:\n\n"
        
        for i, sent in enumerate(top_sentences, 1):
            if sent:
                answer += f"**{i}.** {sent}.\n\n"
        
        if rule_refs:
            answer += f"\n### 📌 Referenced Rules: {', '.join(rule_refs[:5])}\n\n"
        
        answer += "### ⚠️ Important Notes:\n\n"
        answer += "- The above excerpts are directly from your uploaded rulebook(s).\n"
        answer += "- For a more detailed analysis with interpretations and practical applications, add an Anthropic API key in the sidebar.\n"
        answer += "- Always verify rulings with your league's official rulebook and consult with experienced officials for complex situations.\n"
        
        confidence = min(0.7, 0.15 * len(top_sentences))
    else:
        answer = "## 🔍 Search Results\n\n"
        answer += "I found some content in the rulebook but couldn't match it specifically to your question.\n\n"
        answer += "### Related Content:\n\n"
        answer += f"{context[:800]}...\n\n"
        answer += "### 💡 Suggestions:\n\n"
        answer += "- Try rephrasing your question with specific terms from the rulebook\n"
        answer += "- Add an API key for more intelligent rule interpretation\n"
        answer += "- Make sure you've selected the correct sport and competition level\n"
        confidence = 0.3
    
    return {
        "answer": answer,
        "confidence": confidence,
        "source": "fallback"
    }

def answer_question(question: str, category: Optional[str], level: Optional[str], document_id: Optional[str], api_key: Optional[str], conversation_history: List[Dict] = None) -> Dict:
    """Answer a question using RAG with conversation context."""
    # Search for relevant chunks - get more for better analysis
    chunks = search_chunks(question, category, level, document_id, limit=10)
    
    # If this might be a follow-up, also search based on previous context
    if conversation_history and len(conversation_history) >= 2:
        # Get last assistant response to find related content
        last_messages = conversation_history[-4:]  # Last 2 exchanges
        context_keywords = []
        for msg in last_messages:
            if msg["role"] == "user":
                context_keywords.extend(tokenize(msg["content"]))
        
        # Search with combined context
        if context_keywords:
            combined_query = question + " " + " ".join(context_keywords[:10])
            additional_chunks = search_chunks(combined_query, category, level, document_id, limit=5)
            
            # Merge results, avoiding duplicates
            existing_ids = {c["id"] for c in chunks}
            for chunk in additional_chunks:
                if chunk["id"] not in existing_ids:
                    chunks.append(chunk)
    
    if not chunks:
        return {
            "answer": "No relevant documents found. Please upload rulebooks for the selected sport and level.",
            "sources": [],
            "rule_references": [],
            "confidence": 0.0
        }
    
    # Build context with more detail
    context = "\n\n---\n\n".join([
        f"[Source: {c['filename']} | Relevance Score: {c['score']:.2f}]\n{c['content']}" 
        for c in chunks[:10]  # Limit to top 10
    ])
    
    # Generate answer
    if api_key:
        result = generate_answer_with_claude(question, context, api_key, conversation_history)
    else:
        result = generate_fallback_answer(question, context)
    
    # Collect rule references and sources
    all_refs = set()
    sources = []
    for chunk in chunks:
        all_refs.update(chunk.get("rule_references", []))
        if chunk["filename"] not in sources:
            sources.append(chunk["filename"])
    
    return {
        "answer": result["answer"],
        "sources": sources,
        "rule_references": list(all_refs)[:10],
        "confidence": result["confidence"]
    }

# Document management
def get_documents(category: Optional[str] = None, level: Optional[str] = None) -> List[Dict]:
    """Get all documents."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    sql = "SELECT id, filename, category, level, upload_date, file_size, chunk_count FROM documents WHERE 1=1"
    params = []
    
    if category:
        sql += " AND category = ?"
        params.append(category)
    
    if level:
        sql += " AND level = ?"
        params.append(level)
    
    sql += " ORDER BY upload_date DESC"
    
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "filename": row[1],
            "category": row[2],
            "level": row[3],
            "upload_date": row[4],
            "file_size": row[5],
            "chunk_count": row[6]
        }
        for row in rows
    ]

def delete_document(doc_id: str) -> bool:
    """Delete a document and its chunks."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return True
    except:
        conn.rollback()
        return False
    finally:
        conn.close()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Sports Rulebook Q&A",
        page_icon="🏆",
        layout="wide"
    )
    
    # Initialize database
    init_db()
    
    # Create documents directory
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #1a5f4a 0%, #2d8a6e 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .main-header h1 {
            color: white !important;
            margin: 0;
        }
        .main-header p {
            color: #e0e0e0 !important;
            margin: 0.5rem 0 0 0;
        }
        /* Light text for dark backgrounds */
        .stChatMessage [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
        }
        .stChatMessage [data-testid="stCaptionContainer"] p {
            color: #cccccc !important;
        }
        .stMarkdown p, .stMarkdown li, .stMarkdown span {
            color: #f0f0f0 !important;
        }
        /* Make sure all text is visible */
        p, li, span, div {
            color: #f0f0f0;
        }
        /* Conversation list styling */
        .stSidebar [data-testid="stButton"] button {
            text-align: left !important;
            font-size: 0.85rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🏆 Sports Rulebook Q&A</h1>
            <p>Upload rulebooks and get instant answers about sports rules</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
    if "selected_level" not in st.session_state:
        st.session_state.selected_level = None
    if "selected_document" not in st.session_state:
        st.session_state.selected_document = None
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    
    # Sidebar
    with st.sidebar:
        # New Conversation Button
        if st.button("➕ New Conversation", use_container_width=True, type="primary"):
            st.session_state.current_conversation_id = None
            st.session_state.messages = []
            st.session_state.selected_document = None
            st.rerun()
        
        st.divider()
        
        # Conversation History
        st.subheader("💬 Conversations")
        conversations = get_conversations(limit=30)
        
        if conversations:
            for conv in conversations:
                col1, col2 = st.columns([5, 1])
                with col1:
                    # Truncate title for display
                    display_title = conv["title"][:35] + "..." if len(conv["title"]) > 35 else conv["title"]
                    
                    # Highlight current conversation
                    if st.session_state.current_conversation_id == conv["id"]:
                        if st.button(f"📍 {display_title}", key=f"conv_{conv['id']}", use_container_width=True):
                            pass  # Already selected
                    else:
                        if st.button(f"💬 {display_title}", key=f"conv_{conv['id']}", use_container_width=True):
                            # Load this conversation
                            st.session_state.current_conversation_id = conv["id"]
                            st.session_state.messages = get_messages(conv["id"])
                            if conv["category"]:
                                st.session_state.selected_category = conv["category"]
                            if conv["level"]:
                                st.session_state.selected_level = conv["level"]
                            st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del_conv_{conv['id']}", help="Delete conversation"):
                        delete_conversation(conv["id"])
                        if st.session_state.current_conversation_id == conv["id"]:
                            st.session_state.current_conversation_id = None
                            st.session_state.messages = []
                        st.rerun()
        else:
            st.caption("No conversations yet. Start chatting!")
        
        st.divider()
        
        # Settings section
        st.subheader("⚙️ Settings")
        
        # API Key - load saved key
        saved_key = get_api_key() or ""
        api_key = st.text_input(
            "Anthropic API Key", 
            value=saved_key,
            type="password", 
            help="For AI-powered answers. Works without it too!"
        )
        
        # Save button for API key
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Key"):
                if api_key:
                    save_api_key(api_key)
                    st.success("✅ Saved!")
                else:
                    st.warning("Enter a key first")
        with col2:
            if st.button("🗑️ Clear Key"):
                save_api_key("")
                st.success("Cleared!")
                st.rerun()
        
        if saved_key:
            st.caption("✅ API key saved")
        
        st.divider()
        
        # Category selection
        st.subheader("🏅 Select Sport")
        category_options = ["All Sports"] + [f"{c['icon']} {c['name']}" for c in CATEGORIES]
        selected_cat_display = st.selectbox("Sport Category", category_options)
        
        if selected_cat_display == "All Sports":
            st.session_state.selected_category = None
        else:
            for c in CATEGORIES:
                if c["name"] in selected_cat_display:
                    st.session_state.selected_category = c["id"]
                    break
        
        # Level selection
        st.subheader("📊 Select Level")
        level_options = ["All Levels"] + [l["name"] for l in LEVELS]
        selected_lvl_display = st.selectbox("Competition Level", level_options)
        
        if selected_lvl_display == "All Levels":
            st.session_state.selected_level = None
        else:
            for l in LEVELS:
                if l["name"] == selected_lvl_display:
                    st.session_state.selected_level = l["id"]
                    break
        
        # Document selection (optional - filter by specific document)
        st.subheader("📄 Select Document")
        available_docs = get_documents(
            st.session_state.selected_category,
            st.session_state.selected_level
        )
        
        if available_docs:
            doc_options = ["All Documents"] + [doc["filename"] for doc in available_docs]
            selected_doc_display = st.selectbox(
                "Specific Document (optional)", 
                doc_options,
                help="Leave as 'All Documents' to search all, or select a specific rulebook"
            )
            
            if selected_doc_display == "All Documents":
                st.session_state.selected_document = None
            else:
                # Find the document ID
                for doc in available_docs:
                    if doc["filename"] == selected_doc_display:
                        st.session_state.selected_document = doc["id"]
                        break
        else:
            st.caption("No documents available for this selection")
            st.session_state.selected_document = None
        
        st.divider()
        
        # Document stats
        docs = get_documents()
        st.metric("📚 Documents", len(docs))
        st.metric("📄 Total Chunks", sum(d["chunk_count"] for d in docs))
    
    # Navigation
    page = st.radio("", ["💬 Chat", "📤 Upload", "📚 Documents"], horizontal=True)
    
    st.divider()
    
    # Chat Page
    if page == "💬 Chat":
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "metadata" in message:
                    meta = message["metadata"]
                    
                    # Confidence
                    conf = meta.get("confidence", 0)
                    st.caption(f"Confidence: {conf:.0%}")
                    
                    # Sources
                    if meta.get("sources"):
                        st.caption("Sources: " + ", ".join(meta["sources"]))
                    
                    # Rule references
                    if meta.get("rule_references"):
                        st.caption("Rules: " + ", ".join(meta["rule_references"][:5]))
        
        # Chat input (outside of any container)
        if prompt := st.chat_input("Ask a question about sports rules..."):
            # Create new conversation if needed
            if st.session_state.current_conversation_id is None:
                title = generate_title_from_question(prompt)
                st.session_state.current_conversation_id = create_conversation(
                    title=title,
                    category=st.session_state.selected_category,
                    level=st.session_state.selected_level
                )
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            save_message(st.session_state.current_conversation_id, "user", prompt)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching rulebooks..."):
                    result = answer_question(
                        prompt,
                        st.session_state.selected_category,
                        st.session_state.selected_level,
                        st.session_state.selected_document,
                        api_key if api_key else None,
                        st.session_state.messages  # Pass conversation history
                    )
                
                st.markdown(result["answer"])
                
                # Show metadata
                if result["confidence"] > 0:
                    st.caption(f"Confidence: {result['confidence']:.0%}")
                if result["sources"]:
                    st.caption("Sources: " + ", ".join(result["sources"]))
                if result["rule_references"]:
                    st.caption("Rules: " + ", ".join(result["rule_references"][:5]))
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": result
                })
                save_message(
                    st.session_state.current_conversation_id, 
                    "assistant", 
                    result["answer"],
                    result
                )
    
    # Upload Page
    elif page == "📤 Upload":
        st.subheader("📤 Upload Rulebook")
        
        col1, col2 = st.columns(2)
        
        with col1:
            upload_category = st.selectbox(
                "Sport Category",
                options=[c["id"] for c in CATEGORIES],
                format_func=lambda x: next(f"{c['icon']} {c['name']}" for c in CATEGORIES if c["id"] == x)
            )
        
        with col2:
            upload_level = st.selectbox(
                "Competition Level",
                options=[l["id"] for l in LEVELS],
                format_func=lambda x: next(l["name"] for l in LEVELS if l["id"] == x)
            )
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt", "md"],
            help="Supported formats: PDF, Word (.docx), Text (.txt), Markdown (.md)"
        )
        
        if uploaded_file:
            if st.button("📥 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Save file temporarily
                        file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        doc_id = process_document(
                            file_path,
                            uploaded_file.name,
                            upload_category,
                            upload_level
                        )
                        
                        st.success(f"✅ Document processed successfully! ID: {doc_id}")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
    
    # Documents Page
    elif page == "📚 Documents":
        st.subheader("📚 Uploaded Documents")
        
        docs = get_documents(
            st.session_state.selected_category,
            st.session_state.selected_level
        )
        
        if not docs:
            st.info("No documents uploaded yet. Go to the Upload tab to add rulebooks.")
        else:
            for doc in docs:
                with st.expander(f"📄 {doc['filename']}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    # Find category and level names
                    cat_name = next((c["name"] for c in CATEGORIES if c["id"] == doc["category"]), doc["category"])
                    lvl_name = next((l["name"] for l in LEVELS if l["id"] == doc["level"]), doc["level"])
                    
                    col1.write(f"**Category:** {cat_name}")
                    col2.write(f"**Level:** {lvl_name}")
                    col3.write(f"**Chunks:** {doc['chunk_count']}")
                    
                    st.write(f"**Uploaded:** {doc['upload_date'][:10]}")
                    st.write(f"**Size:** {doc['file_size'] / 1024:.1f} KB")
                    
                    if st.button(f"🗑️ Delete", key=f"del_{doc['id']}"):
                        if delete_document(doc["id"]):
                            st.success("Document deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")

if __name__ == "__main__":
    main()
