"""
Chat conversation Memory which manages chat history and context for the follow-up queries.
"""

import sys, json
from typing import List, Dict, Optional
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from system_config import MAX_CHAT_HISTORY
from src.query_classifier.classifier import QueryType 

class ChatConversationHistory:
    """
    This class will manages chat conversation hisoty for context aware responses 
    """
    def __init__(self, max_history_levels: int = MAX_CHAT_HISTORY):
        """
        Initializing the conversation history 
        Args:
            max_history_levels: Maximum number of messages/chat levels to hold back
        """
        self.max_history_levels = max_history_levels
        self.chat_messages: List[Dict] = []         # This is for chat messages to hold
        self.query_types: List[str] = []            # This is for query types for the context 
        self.sources_used: List[List[Dict]] = []    # This is for sources for each response

        self.ROLE = "role"
        self.ROLE_USER = "user"
        self.ROLE_ASSISTANT = "assistant"
        self.CONTENT = "content"

    def add_chat_messages(self, role:str, msg_content:str, query_type: Optional[str] = None, sources: Optional[List[Dict]] = None) -> None:
        """
        THis method will add the chat message to the History
        Args:
            role: the one who's sending messages "user" or "assistant"
            msg_content: Message content
            query_type: Type of query (for the assistant to message better)
            sources : List of sources uses for assistant to message
        """
        message = {
            self.ROLE : role,
            self.CONTENT : msg_content
        }
        self.chat_messages.append(message)

        # Tracking query type and sources for assistant messages
        if role == self.ROLE_ASSISTANT:
            self.query_types.append(query_type or QueryType.GENERAL)
            self.sources_used.append(sources or [])
        
        # Trimming conversation history if it execeeds the limit
        if len(self.chat_messages) > self.max_history_levels:
            # Follow LRU caches technique
            self.chat_messages = self.chat_messages[-self.max_history_levels:]

    def get_all_chat_messages(self) -> List[Dict]:
        """
        This method will get all the messages in the chat conversation history
        Returns:
            List of message dictionaries
        """
        return self.chat_messages.copy()
    
    def get_last_n_messages(self, n: int) -> List[Dict]:
        """
        This method will get the last n messages from chat conversation history
        Args: 
            n: Number of messages to retrieve
        Returns:
            List of the most recent n messages
        """
        return self.chat_messages[-n:] if n< len(self.chat_messages) else self.chat_messages.copy()

    def get_formatted_history(self, n:Optional[int] = None) -> str:
        """
        This method will get formatted conversation history for prompts.
        Args:
            n: Number of recent messages to include
        Returns:
            Formatted conversation string
        """
        messages = self.get_last_n_messages(n) if n else self.chat_messages

        formatted_chats = []
        for m in messages:
            role = m.get(self.ROLE, "unknown")
            content = m.get(self.CONTENT, "")
            if role == self.ROLE_USER:
                formatted_chats.append(f"Customer: {content}")
            elif role == self.ROLE_ASSISTANT:
                formatted_chats.append(f"Assistant: {content}")
            
        return "\n".join(formatted_chats)
    
    def get_last_query_type(self) -> Optional[str]:
        """
        This method will get the query type of the last assistant response.
        Returns:
            Last query type of None
        """
        if self.query_types:
            return self.query_types[-1]
        return None
    
    def get_last_sources_list(self) -> List[Dict]:
        """
        This method will get the sources used in the last assistant response.
        Returns:
            List of source dictionaries
        """
        if self.sources_used:
            return self.sources_used[-1]
        return []
    
    def get_context_for_follow_up(self) -> Dict:
        """
        This method will give the context information to handle follow-up queries.
        Returns:
            Dictionary with context information
        """
        last_user_message = None
        last_assistant_message = None
        last_query_type = None
        last_sources = []

        # Iterate in reverse to find last messages
        for msg in reversed(self.chat_messages):
            if msg[self.ROLE] == self.ROLE_ASSISTANT and last_assistant_message is None:
                last_assistant_message = msg[self.CONTENT]
            elif msg[self.ROLE] == self.ROLE_USER and last_user_message is None:
                last_user_message = msg[self.CONTENT]
            
            if last_user_message and last_assistant_message:
                break

        if self.query_types:
            last_query_type = self.query_types[-1]
        if self.sources_used:
            last_sources = self.sources_used[-1]

        return {
            "last_user_message": last_user_message,
            "last_assistant_message": last_assistant_message,
            "last_query_type": last_query_type,
            "last_sources":last_sources,
            "conversation_length": len(self.chat_messages)
        }
    
    def clear(self) -> None:
        """ Clear all the chat conversation messages from the history. """
        self.chat_messages = []
        self.query_types = []
        self.sources_used = []

    def to_dict(self) -> Dict:
        """
        Converts all the messages in the history to dictionary for serialization.
        Returns:
            Dictionary representation of all chat messages
        """
        
        return {
            "messages": self.chat_messages,
            "query_types": self.query_types,
            "sources_used": self.sources_used,
            "max_history": self.max_history_levels
        }
    
    @classmethod
    def from_dict(cls, data:Dict) -> "ChatConversationHistory":
        """
        This method will create ChatConversationHistory from the input dictionary 
        Args: 
            data: input data in dictionary form
        Returns:
            ChatConversationHistory instance
        """
        chat_memory = cls(max_history_levels=data.get("max_history", MAX_CHAT_HISTORY))
        chat_memory.chat_messages = data.get("messages", [])
        chat_memory.query_types = data.get("query_types", [])
        chat_memory.sources_used = data.get("sources_used", [])
        
        return chat_memory
    
    def save_chat_history_to_file(self, filepath: str) -> None:
        """
        Saves chat conversation memory to file
        ARgs:
            filepath : File path to save data into
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_chat_history_from_file(cls, filepath: str) -> "ChatConversationHistory":
        """
        Loads chat conversation history from the file.
        Args; filepath : file path to load the data from
        Returns: ChatConversationHistory object 
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)
    
    def __len__(self) -> int:
        """ This will return total number of messages in memory """
        return len(self.chat_messages)
    
    def __representation_of_chats__(self) -> str:
        """ String representation of chat message in the memory """
        return f"ChatConversationHistory(chat_messages={len(self.chat_messages)}, max_history_levels= {self.max_history_levels})"






