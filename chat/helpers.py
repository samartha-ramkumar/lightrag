import json
import datetime
from bson import ObjectId
from fastapi import HTTPException

# In-memory storage for chat sessions
CHAT_SESSIONS = {}

async def chat_session_exists(session_id):
    """
    Checks if a chat session exists and 
    raises an HTTP exception if not found.

    :param session_id: The session ID to check.
    :return: The session document if found.
    :raises HTTPException: If the session does not exist.
    """
    try:
        if session_id not in CHAT_SESSIONS:
            raise HTTPException(status_code=404, detail="Session not found.")
        
        return CHAT_SESSIONS[session_id]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def append_message_to_chat_history(session_id, message_data):
    """
    Appends a new message to the chat history for a given session. 
    Accepts a dictionary of message data in OpenAI format.
    Expected format: {"role": "user|assistant", "content": "message"}
    """
    session = await chat_session_exists(session_id)
    
    # Add timestamp to message data 
    message_data["timestamp"] = datetime.datetime.now().isoformat()
    
    try:
        session["messages"].append(message_data)
            
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

async def get_messages_for_session(session_id):
    """
    Retrieves the list of messages for a given chat session.
    """
    session = await chat_session_exists(session_id)
    
    messages = session.get("messages", [])
    
    # Sort by timestamp if available
    messages.sort(key=lambda x: x.get("timestamp", ""))
    
    return messages

async def create_new_chat_session():
    """
    Creates a new chat session.
    """
    session_id = str(ObjectId())
    
    try:
        document = { 
            "session_id": session_id,
            "messages": [], 
            "created_at": datetime.datetime.now().isoformat(),
        }
        
        CHAT_SESSIONS[session_id] = document

    except Exception as e:
        raise Exception(str(e))

    return session_id

# New functions for LightRAG conversation management
def validate_message_format(message):
    """Validate that a message has the correct format"""
    if not isinstance(message, dict):
        return False
    if "role" not in message or "content" not in message:
        return False
    if message["role"] not in ["user", "assistant", "system"]:
        return False
    if not isinstance(message["content"], str):
        return False
    return True

def validate_conversation_messages(messages):
    """Validate a list of conversation messages"""
    if not isinstance(messages, list):
        return False
    return all(validate_message_format(msg) for msg in messages)

def add_message_to_conversation(messages, role, content):
    """Add a new message to conversation history"""
    if role not in ["user", "assistant", "system"]:
        raise ValueError(f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'")
    
    new_messages = messages.copy()
    message = {"role": role, "content": content}
    new_messages.append(message)
    return new_messages

def get_chat_conversation_turns(conversation_history, max_turns=None):
    """Convert conversation history to context string with optional turn limit"""
    if not conversation_history:
        return ""
    
    # If max_turns is specified, get only the last N turns
    if max_turns and max_turns > 0:
        conversation_history = conversation_history[-max_turns * 2:]  # 2 messages per turn (user + assistant)
    
    context_parts = []
    for msg in conversation_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            context_parts.append(f"{role.capitalize()}: {content}")
    
    return "\n".join(context_parts)

def get_last_n_messages(messages, n):
    """Get the last n messages from conversation"""
    if n <= 0:
        return []
    return messages[-n:] if len(messages) > n else messages


def conversation_to_openai_format(messages):
    """Convert conversation messages to OpenAI API format"""
    formatted_messages = []
    
    for message in messages:
        if validate_message_format(message):
            formatted_message = {
                "role": message["role"],
                "content": message["content"]
            }
            formatted_messages.append(formatted_message)
    
    return formatted_messages

def merge_conversation_contexts(conversation_history, system_context=None):
    """Merge conversation history with system context"""
    messages = []
    
    if system_context:
        messages.append({"role": "system", "content": system_context})
    
    messages.extend(conversation_history)
    return messages

## EXAMPLE USAGE
if __name__ == "__main__":
    import asyncio
    
    async def test_chat_helpers():
        """Example usage with multiple chat sessions and conversations."""
        
        session1_id = await create_new_chat_session()

        print(f"   Created sessions: {session1_id}\n")
        
        # Session 1: Python Programming Conversation
        print("2. Session 1 - Python Programming Conversation...")
        python_messages = [
            {"role": "user", "content": "Hello, can you help me with Python programming?"},
            {"role": "assistant", "content": "Of course! I'd be happy to help you with Python programming. What specific topic would you like to learn about?"},
            {"role": "user", "content": "Can you explain what async/await is in Python?"},
            {"role": "assistant", "content": "Async/await in Python is used for asynchronous programming. The 'async' keyword defines a coroutine function, and 'await' is used to pause execution until an awaitable object completes."},
            {"role": "user", "content": "Can you show me a simple example?"},
            {"role": "assistant", "content": "Sure! Here's a simple example:\n\n```python\nimport asyncio\n\nasync def fetch_data():\n    await asyncio.sleep(1)\n    return 'Data fetched!'\n\nasync def main():\n    result = await fetch_data()\n    print(result)\n\nasyncio.run(main())\n```"}
        ]
        
        for msg in python_messages:
            await append_message_to_chat_history(session1_id, msg)
        

        # Display all conversations
        print("5. Displaying all conversations...\n")
        
        sessions_info = [
            (session1_id, "Python Programming"),
        ]
        
        for session_id, topic in sessions_info:
            print(f"=== {topic} (Session: {session_id[:8]}...) ===")
            messages = await get_messages_for_session(session_id)
            print(f"messages: {messages}")
        
        # Test session existence
        print("6. Testing session operations...")
        for session_id, topic in sessions_info:
            session = await chat_session_exists(session_id)
            print(f"   {topic}: {len(session['messages'])} messages")
        

        print("7. Testing non-existent session retrieval...")
        try:
            await get_messages_for_session("non_existent_session_id")
        except HTTPException as e:
            print(f"   Error: {e.detail}")

        # Test new LightRAG conversation functions
        print("8. Testing LightRAG conversation functions...")
        
        # Test message validation
        valid_msg = {"role": "user", "content": "Hello"}
        invalid_msg = {"role": "invalid", "content": "Hello"}
        
        print(f"   Valid message validation: {validate_message_format(valid_msg)}")
        print(f"   Invalid message validation: {validate_message_format(invalid_msg)}")
        
        # Test adding messages
        conv_messages = []
        conv_messages = add_message_to_conversation(conv_messages, "user", "Hello")
        conv_messages = add_message_to_conversation(conv_messages, "assistant", "Hi there!")
        
        print(f"   Conversation after adding messages: {len(conv_messages)} messages")
        
        # Test getting last n messages
        last_2 = get_last_n_messages(conv_messages, 2)
        print(f"   Last 2 messages: {len(last_2)} messages")
        
        # Test conversation to OpenAI format
        openai_format = conversation_to_openai_format(conv_messages)
        print(f"   OpenAI format: {openai_format}")
        
        # Test merging conversation contexts
        merged_context = merge_conversation_contexts(conv_messages, "You are a helpful assistant.")
        print(f"   Merged context: {merged_context}")
        
        # Test getting conversation turns
        turns = get_chat_conversation_turns(conv_messages, max_turns=1)
        print(f"   Conversation turns: {turns}")
    
    # Run the example
    asyncio.run(test_chat_helpers())
