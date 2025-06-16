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
    
    # Run the example
    asyncio.run(test_chat_helpers())
