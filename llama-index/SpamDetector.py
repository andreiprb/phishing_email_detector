import os
from dotenv import load_dotenv
from typing import Tuple, Optional, List
from llama_index.core import load_index_from_storage, StorageContext, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.schema import NodeWithScore

load_dotenv()


class SpamDetector:
    """A robust spam detector using LlamaIndex vector stores with careful prompt engineering."""

    def __init__(
            self,
            llm: Optional[BaseLLM] = None,
            embed_model: Optional[BaseEmbedding] = None,
            spam_index_path: str = "../data/index_spam",
            ham_index_path: str = "../data/index_ham",
            top_k: int = 5
    ):
        """
        Initialize the SpamDetector with LLM and vector stores.

        Args:
            spam_index_path: Path to the spam vector store
            ham_index_path: Path to the ham vector store
            llm: LLM instance (default: Groq with temperature 0)
            top_k: Number of examples to retrieve from each index
        """
        if llm is None:
            api_key = os.getenv("GROQ_API_KEY")
            assert api_key is not None, "GROQ_API_KEY environment variable not set."
            self.llm = Groq(model="gemma2-9b-it", temperature=0, api_key=api_key)
        else:
            self.llm = llm

        if embed_model is None:
            self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        else:
            self.embed_model = embed_model

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.top_k = top_k
        self.spam_index = self._load_index(spam_index_path)
        self.ham_index = self._load_index(ham_index_path)

    def _load_index(self, index_path: str):
        """Load an index from storage."""
        if os.path.exists(index_path):
            storage_context = StorageContext.from_defaults(persist_dir=index_path)
            return load_index_from_storage(storage_context)
        else:
            raise FileNotFoundError(f"Index not found at {index_path}")

    def _extract_key_features(self, nodes: List[NodeWithScore], category: str) -> str:
        """
        Extract key features from the examples instead of using full text.
        This dramatically reduces prompt size while preserving signal.
        """
        if not nodes:
            return f"No {category} examples found."

        features = []
        for i, node in enumerate(nodes[:3]):  # Limit to top 3 for brevity
            text = node.node.text

            # Extract key features based on category
            if category == "spam":
                # For spam, look for suspicious patterns
                indicators = []
                if "http" in text or "www." in text or "click" in text.lower():
                    indicators.append("contains links")
                if "urgent" in text.lower() or "limited time" in text.lower() or "act now" in text.lower():
                    indicators.append("urgency tactics")
                if "congratulation" in text.lower() or "winner" in text.lower() or "selected" in text.lower():
                    indicators.append("unsolicited rewards")
                if "dear customer" in text.lower() or "valued customer" in text.lower():
                    indicators.append("generic greeting")
                if "$" in text or "€" in text or "money" in text.lower() or "cash" in text.lower():
                    indicators.append("money-related")

                if not indicators:
                    # Fallback if no patterns detected
                    snippet = text[:100] + "..." if len(text) > 100 else text
                    indicators.append(f"sample text: {snippet}")

                features.append(f"SPAM EXAMPLE {i + 1}: {', '.join(indicators)}")
            else:
                # For ham, focus on legitimate patterns
                indicators = []
                if "@" in text and not ("http" in text or "www." in text):
                    indicators.append("personal communication")
                if len(text.split()) > 20 and text.count('.') > 3:
                    indicators.append("detailed content")
                if "meeting" in text.lower() or "schedule" in text.lower() or "project" in text.lower():
                    indicators.append("business/personal context")
                if "thanks" in text.lower() or "thank you" in text.lower():
                    indicators.append("personal acknowledgment")

                if not indicators:
                    # Fallback if no patterns detected
                    snippet = text[:100] + "..." if len(text) > 100 else text
                    indicators.append(f"sample text: {snippet}")

                features.append(f"HAM EXAMPLE {i + 1}: {', '.join(indicators)}")

        return "\n".join(features)

    def classify(self, email_text: str) -> Tuple[str, float]:
        """
        Classify an email as spam or ham using vector retrieval + LLM.

        Args:
            email_text: The text content of the email to classify

        Returns:
            Tuple containing classification ('spam' or 'ham') and confidence score
        """
        # Get similar examples from both indices
        spam_retriever = self.spam_index.as_retriever(similarity_top_k=self.top_k)
        ham_retriever = self.ham_index.as_retriever(similarity_top_k=self.top_k)

        spam_nodes = spam_retriever.retrieve(email_text)
        ham_nodes = ham_retriever.retrieve(email_text)

        # Calculate relevance scores (higher = more similar to input)
        avg_spam_score = sum(node.score for node in spam_nodes) / len(spam_nodes) if spam_nodes else 0
        avg_ham_score = sum(node.score for node in ham_nodes) / len(ham_nodes) if ham_nodes else 0

        # Extract key features instead of full examples
        spam_features = self._extract_key_features(spam_nodes, "spam")
        ham_features = self._extract_key_features(ham_nodes, "ham")

        # Define common spam indicators explicitly
        spam_indicators = """
        SPAM INDICATORS:
        - Unsolicited offers, rewards, or prizes
        - Urgent calls to action ("Act now", "Limited time")
        - Suspicious links or attachments
        - Generic greetings ("Dear Customer")
        - Too good to be true offers
        - Requests for personal information
        - Excessive punctuation or ALL CAPS
        - Misspellings or poor grammar
        - Suspicious sender addresses
        - Requests to click links or download files
        """

        # Define common ham indicators explicitly
        ham_indicators = """
        HAM INDICATORS:
        - Expected or solicited communication
        - Specific and personalized content
        - Relevant to recipient's work, personal life, or interests
        - Professional or personal tone appropriate to the relationship
        - Specific greeting with recipient's name
        - Coherent and logical content
        - Proper grammar and formatting
        - Clear sender identification
        """

        # Build a shorter, more focused prompt
        prompt = f"""You are an expert email spam detector. Analyze this email and determine if it's spam or legitimate (ham).

        {spam_indicators}
        {ham_indicators}

        SIMILARITY SCORES:
        - Similarity to known spam: {avg_spam_score:.4f}
        - Similarity to known ham: {avg_ham_score:.4f}

        SIMILAR SPAM EXAMPLES:
        {spam_features}

        SIMILAR HAM EXAMPLES:
        {ham_features}

        EMAIL TO CLASSIFY:
        {email_text}

        First, identify any spam indicators in the email.
        Then, identify any ham indicators in the email.
        Finally, determine if this is "spam" or "ham" based on the indicators and provide your confidence (0.0-1.0).

        CLASSIFICATION:"""

        # Get LLM response
        response = self.llm.complete(prompt)
        response_text = response.text.strip()

        # Parse the response
        try:
            # Look for explicit classification statement
            if "spam" in response_text.lower() and "ham" in response_text.lower():
                # Both words appear, need to determine which is the conclusion
                lines = response_text.lower().split('\n')
                last_few_lines = ' '.join(lines[-3:])  # Check the final conclusion

                classification = "spam" if last_few_lines.count("spam") > last_few_lines.count("ham") else "ham"

                # Extract confidence if present
                import re
                confidence_matches = re.findall(r'confidence[:\s]+([0-9.]+)', response_text.lower())
                confidence = float(confidence_matches[0]) if confidence_matches else 0.8
            else:
                # Simpler case - one clearly dominates
                classification = "spam" if response_text.lower().count("spam") > response_text.lower().count(
                    "ham") else "ham"

                # Extract confidence if present
                import re
                confidence_matches = re.findall(r'confidence[:\s]+([0-9.]+)', response_text.lower())
                confidence = float(confidence_matches[0]) if confidence_matches else 0.8

            # Special rule - these are dead giveaways for spam
            spam_phrases = ["click here", "congratulations", "limited time offer", "act now", "special offer",
                            "prize", "claim your", "suspicious-link", "suspicious link"]

            has_spam_phrases = any(phrase in email_text.lower() for phrase in spam_phrases)
            has_urls = "http" in email_text or "www." in email_text

            if has_spam_phrases and has_urls:
                # Override for obvious spam
                classification = "spam"
                confidence = max(confidence, 0.9)  # Boost confidence for obvious spam

            return classification, min(max(confidence, 0.0), 1.0)  # Ensure confidence is between 0 and 1

        except Exception as e:
            # Fallback if parsing fails
            print(f"Error parsing response: {e}")
            # Count occurrences as a simple fallback method
            spam_count = response_text.lower().count("spam")
            ham_count = response_text.lower().count("ham")

            if spam_count > ham_count:
                return "spam", 0.7
            else:
                return "ham", 0.7
