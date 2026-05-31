"""
Query Classifier Module Classifies user queries into types: Troubleshooting, Comparison, or General.
"""

import re, sys
from enum import Enum 
from typing import Optional, Tuple, List 
from pathlib import Path 

sys.path.append(str(Path(__file__).parent.parent.parent))

from system_config import CLASSIFICATION_KEYWORDS

# Enumerations of query types;
class QueryType(Enum):
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    GENERAL = "general"
    FOLLOW_UP = "follow_up"

class SamsungQueryClassifier:
    """
    This class with will be used to classify the use queries into different types of routing.
    This will support both rule-based and LLM-based classification.
    """

    def __init__(self, llm_client=None, is_llm_fallback_needed:bool = True):
    
        """
        This constructor is used to initialize the query classifier.
        Args:
            llm client: LLM client used for LLM-based classification 
            is llm fallback_needed: Boolean whether to use LLM for ambiguous cases also
        """

        self.llm_client = llm_client
        self.is_llm_fallback_needed = is_llm_fallback_needed
        self.keywords = CLASSIFICATION_KEYWORDS
        
        #regex based patterns for rule based classiciation
        self.patterns = {
            QueryType.TROUBLESHOOTING: [
                r"why is my .+ (overheating|not working|broken|slow)", 
                r"how (do i|to) (fix|reset|troubleshoot|repair)", 
                r"my .+ (won't|will not|doesn't|does not) (turn on|work|charge)", 
                r"(error|issue|problem) with my", 
                r"what('s| is) wrong with", 
                r"help (me )?(fix|with|troubleshoot)",
            ],
            QueryType.COMPARISON: [
                r"compare .+ (vs|versus|and) .+", 
                r"difference(s)? between .+ and .+", 
                r"which (one )?(is )?(better|best|should i)", 
                r".+V5 .+", 
                r".t versus .+",
            ], 
            QueryType.GENERAL: [ 
                r"what is", 
                r"how (do(es)?|to|can I)", 
                r"explain", 
                r"tell me about",
                r"tell me about", 
                r"describe",
                r"what are",
            ]
        }

        #Follow-up indicators
        self.follow_up_indicators = [
            "what about", "how about", "is still", "it still", "and the", "this keeps", "still not", 
            "what if", "that didn't", "also", "additionally", "furthermore", "moreover"
        ]
    
    def is_follow_up_query(self, query: str, conversation_history: Optional[List[dict]]) -> bool:
        """
        Check whether the query is a follow-up to the previous conversation.

        Args:
            query: The query string
            conversation_history: Previous messages

        Returns:
            True if the query is follow-up else False
        """

        if not conversation_history or len(conversation_history) == 0:
            return False
        
        # Check for the follow-up indicators
        for indicator in self.follow_up_indicators:
            if indicator in query:
                return True
        
        # If the query is very short, then it can be likely a followup
        # But only if it doesn't match any strong classification keywords/patters etc.,
        if len(query.split()) <= 3:
            # Check if the query matches any troubleshooting or comparison keywords first
            for query_type, keywords in self.keywords.items():
                if any(kw in query for kw in keywords):
                    return False
            #Check if the query matches any patterns..
            for query_type, patterns in self.patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query):
                        return False
            if conversation_history and conversation_history[-1].get("role") == "assistant":
                return True
            
        return False

    def classify(self, query:str, conversation_history: Optional[List[dict]] = None) -> Tuple[QueryType, float, str]:
        """
        This method will classify a query into one of the predefined types.

        Args: 
            query: User's query in the form of string
            conversation_history : If there is any conversation message history we use it for the context, else None

        Returns:
            A tuple consisting of query type, confidence score, reasoning string
        """

        query_in_lower = query.lower().strip()

        # Check whether this is the follow up query or not first
        if self.is_follow_up_query(query_in_lower, conversation_history):
            return QueryType.FOLLOW_UP, 0.9, "Follow-up to the previous conversation"
        
        # 1. Try rule based classification first
        rule_query_type, rule_confidence, rule_reasoning = self.call_rule_based_classification(query_in_lower)

        # 2. If the confidence score is low, LLM fallback is enabled and uses LLM to classify
        if rule_confidence < 0.7 and self.is_llm_fallback_needed and self.llm_client:
            llm_query_type, llm_confidence, llm_reasoning = self.call_llm_based_classification(query_in_lower)
            if llm_confidence > rule_confidence:
                print("-"*80)
                print(f"llm query type:{rule_query_type}\n llm confidence:{rule_confidence}\n llm reasoning:{rule_reasoning}")
                print("-"*80)

                return llm_query_type, llm_confidence, llm_reasoning
        print("-"*80)
        print(f"query type:{rule_query_type}\n confidence:{rule_confidence}\n reasoning:{rule_reasoning}")
        print("-"*80)
        return rule_query_type, rule_confidence, rule_reasoning
    
    def call_rule_based_classification(self, query:str) -> Tuple[QueryType, float, str]:
        """
        This method performs rule-based classification using keywords and patterns

        Args:
            query: lower case query string

        Returns:
            Tuple of (Query type, confidence score, reasoning string)
        """

        scores = {QueryType.TROUBLESHOOTING:0.0, QueryType.COMPARISON:0.0, QueryType.GENERAL:0.0}

        reasoning_parts = []

        # Basic keyword matching logic
        for query_type, keywords in self.keywords.items():
            matched_keywords = [kw for kw in keywords if kw in query]
            if matched_keywords:
                # we score based on number of matched keywords
                scores[QueryType(query_type)] = len(matched_keywords) * 0.3
                reasoning_parts.append(f"Matched {query_type} keywords: {matched_keywords}")
        
        # Pattern matching logic
        for query_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    scores[query_type] += 0.4
                    reasoning_parts.append(f"Matched {query_type.value} pattern: {pattern}")
        
        # check and get the highest type
        best_type = max(scores, key=scores.get)

        # check and get the highest score
        best_score = min(scores[best_type], 0.95) # max will keep to 0.95

        # If there is no strong match then keep the values default to general
        if best_score < 0.2:
            return QueryType.GENERAL, 0.5, "Default to General (No strong match found.)"
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Keyword and pattern matching"

        return best_type, best_score, reasoning
    
    def call_llm_based_classification(self, query=str) -> Tuple[QueryType, float, str]:
        """
        This method helps in LLM based classification of query
        Args: 
            query: The query string
        Returns:
            Tuple of (QueryType, confidence, reasoning) 
        """

        if not self.llm_client:
            return QueryType.GENERAL, 0.5, "LLM is not available"
        
        classification_prompt = f"""
        Classify the following query into exactly one of these categories:
        - troubleshooting: Issues, problems, errors, fixes, repairs
        - comparison: Comparing products, features, differences between items
        - general : General information, how-to questions, explanations

        Query: "{query}"

        Respond in this EXACT format:
        Category: [troubleshooting/comparison/general]
        Confidence: [0.0 - 1.0]
        Reasoning: [Brief explanation in a line]
        """

        try:
            response = self.llm_client.invoke(classification_prompt)
            res_message = response.content if hasattr(response, "content") else str(response)

            # Logic to parse the response
            query_type = QueryType.GENERAL
            confidence = 0.5
            reasoning = "LLM classification"

            for line in res_message.split("\n"):
                if "Category:" in line:
                    category = line.split(":")[1].strip().lower()
                    if "troubleshooting" in category:
                        query_type = QueryType.TROUBLESHOOTING
                    elif "comparison" in category:
                        query_type = QueryType.COMPARISON
                elif "Confidence:" in line:
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except ValueError:
                        confidence = 0.7
                elif "Reasoning:" in line:
                    reasoning = line.split(":")[1].strip()
                
            return query_type, confidence, reasoning
        except Exception as e:
            return QueryType.GENERAL, 0.5, f"LLM classification failed: {str(e)}"
    
    # Utility methods to show type of query with streamlit UI
    def get_query_type_description_for_GUI(self, query_type: QueryType) -> str:
        """
        Get the description of query type
        Args: 
            query_type : Query Type enum value
        Returns:
            description of query type
        """
        descriptions = {
            QueryType.TROUBLESHOOTING : "Resolving technical issues step by step",
            QueryType.COMPARISON : "Comparative analysis with tabular insights and recommendations",
            QueryType.GENERAL: "Clear explanations and informative responses",
            QueryType.FOLLOW_UP : "Follow-up responses based on conversation history and previous context"
        }
        return descriptions.get(query_type, "General Description")
    
    def get_query_type_name_for_GUI(self, query_type:QueryType) -> str:
        """
        Presentable text for query type to show in GUI
        Args:
            query_type : type of the query
        Returns:
            readable query type string to show in GUI
        """
        names = {
            QueryType.TROUBLESHOOTING : "🛠️ Troubleshooting",
            QueryType.COMPARISON : "🧮 Comparison",
            QueryType.GENERAL: "❓ General Query",
            QueryType.FOLLOW_UP : "🔄 Follow-up"
        }
        return names.get(query_type, "❓ General Query")