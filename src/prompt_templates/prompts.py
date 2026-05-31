"""
This will provide structured prompt templates for different query types
"""

import sys
from typing import Optional
from pathlib import Path
from src.query_classifier.classifier import QueryType

sys.path.append(str(Path(__file__).parent.parent.parent))

class SupportPromptTemplates:
    """
    This class will manages the prompt templates for different query types
    Each query type has different prompt templates for different set of output
    """

    # base system context
    SYSTEM_CONTEXT = """You are a helful Samsung Support Assistant.
     You provide accurate, helpful, and friendly support for Samsung products which includes SmartPhones, Tablets, TVs and Applicances.
     Your responses are always based on the provided context from the knowledge base.
     If the context doesn't contain relevant information, acknowledge this and provide general guidance while nothing the limitations."""

    @staticmethod
    def get_prompt_for_troubleshooting_query(query: str, context: str, conversation_history: Optional[str] = None) -> str:
        """
        This helps with prompt for Troubleshooting queries.
        Args: query:User's query, context: Retrieved context from knowledge base, conversation_history: Previous conversation history context
        Returns: Formatted prompt string
        """ 

        history_conv = f"\n\nPrevious Conversation:\n{conversation_history}\n" if conversation_history else ""

        prompt = f"""{SupportPromptTemplates.SYSTEM_CONTEXT} 
                You are helping a customer with a TROUBLESHOOTING issue. Provide a structured, step-by-step response.
                {history_conv}

                KNOWLEDGE BASE CONTEXT:
                {context}

                USER QUERY:
                {query}

                IMPORTANT : Provide your reponse in the EXACT below format:

                🛠️ TROUBLESHOOTING GUIDE
                
                🛠️ POSSIBLE CAUSES:
                1. [Cause 1]
                2. [Cause 2]
                3. [Cause 3]

                🛠️ STEP-BY-STEP SOLUTION: 
                Step 1: [Clear action with details]
                Step 2: [Clear action with details]
                Step 3: [Clear action with details]
                (Add more steps as needed)

                🛠️ WHEN TO ESCALATE:
                - [Condition 1 for esclation]
                - [Condition 2 for esclation]

                🛠️ SOURCE:
                [Document name(s) used for this response or "General troubleshooting guidance" if no specific document found]

                Guidelines:
                - Be specfic and actionable in each step
                - Include any relevant button combinations, settings or procedures
                - If the context doesn't fully address the issue, note what additional information might help 
                """
        return prompt

    @staticmethod
    def get_prompt_for_comparison_query(query:str, context:str, conversation_history: Optional[str] = None) -> str:
        """ This helps with prompt for Compariosn queries"""
        history_conv = f"\n\nPrevious Conversation:\n{conversation_history}\n" if conversation_history else ""

        prompt = f"""{SupportPromptTemplates.SYSTEM_CONTEXT}
                You are helping a customer to COMPARE products. Provide a structured comparison with a clear table format.

                {history_conv}

                KNOWLEDGE BASE CONTEXT:
                {context}

                USER QUERY:
                {query}

                IMPORTANT: Provide your response in the EXACT below format:

                ⚖️ PRODUCT COMPARISON

                |   Feature   |    Product A    |   Product B   |
                |-------------|-----------------|---------------|
                | [Feature 1] |     [Value]     |   [Value]     |
                | [Feature 2] |     [Value]     |   [Value]     |
                | [Feature 3] |     [Value]     |   [Value]     |
                | [Feature 4] |     [Value]     |   [Value]     |
                | [Feature 5] |     [Value]     |   [Value]     |

                (Replace "Product A" and "Product B" with actual product names from the query)

                ⚖️ KEY DIFFERENCES:
                - [Difference 1 with the impact]
                - [Difference 2 with the impact]
                - [Difference 3 with the impact]

                ⚖️ RECOMMENDATION:
                [Clear recommendation based on the comparison, considering different user needs]

                ⚖️ SOURCE:
                [Document name(s) used for this response]

                Guidelines:
                - Include at least 5-7 relevant features in the comparison table
                - Focus on meaningful differences that affect user experience
                - Consider different user profiles (casual user, power user, etc.,) in recommendation
                - If information is incomplete, note what couldn't be compared"""

        return prompt
    
    @staticmethod
    def get_prompt_for_general_query(query: str, context: str, conversation_history: Optional[str] = None) -> str:
        """This helps with prompt for General queries"""
        history_conv = f"\n\nPrevious Conversation:\n{conversation_history}\n" if conversation_history else ""

        prompt = f"""{SupportPromptTemplates.SYSTEM_CONTEXT}
                You are helping a customer with a GENERAL question. Provide a clear, informative response.

                {history_conv}

                KNOWLEDGE BASE CONTEXT:
                {context}

                USER QUERY:
                {query}

                IMPORTANT: Provide your response in the EXACT format below:

                💬 DIRECT ANSWER:
                [Clear, conside answer to the question - 1-2 sentences]

                💬 EXPLANATION:
                [Detailed explanation with relevant information]

                💬 ADDITIONAL NOTES:
                - [Note 1 - tips, warnings or related information]
                - [Note 2 if applicable]
                - [Note 3 if applicable]

                💬 SOURCE:
                [Document name(s) used for this response or "General knowledge" if no specific document found]

                Guidelines:
                - Start with a direct, clear answer
                - Provide enough detail in the explanation for understanding
                - Include practical tips in additional notes
                - Keep the response focused and relevant"""
        
        return prompt
    
    @staticmethod
    def get_prompt_for_follow_up_queries(query:str, context:str, conversation_history: str) -> str:
        """ This helps with prompt for follow-up queries """

        prompt = f"""{SupportPromptTemplates.SYSTEM_CONTEXT}

                You are continuing a conversation with a customer. Consider the previous context when responding.

                PREVIOUS CONVERSATION:
                {conversation_history}

                KNOWLEDGE BASE CONTEXT:
                {context}

                FOLLOW-UP QUERY:
                {query}

                IMPORTANT:
                1. Reference the previous conversation context when relevant
                2. If this is a follow-up to a troubleshooting issue, continue with the same structured format
                3. If this is a new topic, use the appropriate format (troubleshooting/comparison/general)
                4. Avoid repeating information already provided unless clarification is needed

                Provide your response in the appropriate structured format based on the query type."""
        
        return prompt
    
    @staticmethod
    def get_prompt_based_on_query_type(query_type: QueryType, query: str, context: str, 
                                       conversation_history: Optional[str] = None) -> str:
        """ This helps in getting the prompt template based on query type"""

        if query_type == QueryType.TROUBLESHOOTING:
            return SupportPromptTemplates.get_prompt_for_troubleshooting_query(query, context, conversation_history)
        elif query_type == QueryType.COMPARISON:
            return SupportPromptTemplates.get_prompt_for_comparison_query(query, context, conversation_history)
        elif query_type == QueryType.FOLLOW_UP:
            follow_up_history = conversation_history or "No previous conversation available"
            return SupportPromptTemplates.get_prompt_for_follow_up_queries(query, context, follow_up_history)
        else:
            return SupportPromptTemplates.get_prompt_for_general_query(query, context, conversation_history)
        
    @staticmethod
    def get_prompt_when_no_context_available(query: str, query_type: QueryType) -> str:
        """ This helps in getting the prompt when there is no relevant context found in the knowledge base"""

        prompt = f"""{SupportPromptTemplates.SYSTEM_CONTEXT}

                A customer has asked a question, but no relevant information was found in the knowledge base.

                USER QUERY:
                {query}

                QUERY TYPE:
                {query_type.value}

                As there is no relevant information is available in the knowledge base:
                1. Acknowledge that specific information is not available in the current documentation
                2. Provide general guidance if you can do so safely
                3. Suggest the user to contact Samsung Support directly for accurate information
                4. Do NOT make up specific product details or procedures

                Keep your response helpful but honest about the limitations. """
        
        return prompt
    
    @staticmethod
    def get_formatted_conversation_history(messages: list) -> str:
        """ This helps in formatting conversation history for inclusion in prompts """
        formatted = []
        for msg in messages:
            role = msg.get("role", "Unknown")
            content = msg.get("content", "")
            
            if role == "user":
                formatted.append(f"Customer: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")

        return "\n".join(formatted)
