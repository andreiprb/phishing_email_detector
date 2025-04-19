"""
Email Spam Detector using LlamaIndex
-----------------------------------
This script implements an email spam detector using LlamaIndex and a language model.
"""

import os
import email
import logging
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Setup LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
from llama_index.llms.groq import Groq
from llama_index.core.response import Response
from llama_index.core.evaluation import ResponseEvaluator
import pandas as pd

load_dotenv()
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailSpamDetector:
    """Email spam detector using LlamaIndex and LLMs."""

    def __init__(
            self,
            model_name: str = "gemma2-9b-it",
            spam_training_dir: Optional[str] = None,
            temperature: float = 0.1
    ):
        """
        Initialize the spam detector.

        Args:
            model_name: The name of the LLM model to use
            spam_training_dir: Directory containing spam training emails
            temperature: Temperature for LLM generation
        """
        self.llm = Groq(model=model_name, temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))

        # Initialize with training data if provided
        if spam_training_dir:
            self.train(spam_training_dir)
        else:
            self.index = None

        self.evaluator = self._create_spam_evaluator()

    def _create_spam_evaluator(self) -> ResponseEvaluator:
        """Create a response evaluator for spam classification."""
        eval_prompt = """
        You are a professional email spam evaluator with expertise in identifying phishing, 
        scams, and unwanted promotional content. Analyze the email below and determine 
        if it is spam or not.

        Email: {query}

        Consider the following factors:
        - Unsolicited offers or deals that are too good to be true
        - Requests for personal information or financial details
        - Poor grammar, unusual formatting, or excessive urgency
        - Suspicious links or attachments
        - Sender's email domain doesn't match the claimed organization
        - Excessive use of marketing language or pressure tactics

        Provide your final verdict: Is this email spam? Answer with "spam" or "not spam" only.
        """

        return ResponseEvaluator(llm=self.llm, eval_template=eval_prompt)

    def train(self, training_dir: str) -> None:
        """
        Train the detector with example emails.

        Args:
            training_dir: Directory containing training emails
        """
        logger.info(f"Loading training data from {training_dir}")
        documents = SimpleDirectoryReader(training_dir).load_data()

        # Create vector store index
        self.index = VectorStoreIndex.from_documents(
            documents,
            service_context=self.service_context
        )
        logger.info("Training completed")

    def _extract_email_content(self, email_path: str) -> str:
        """
        Extract content from an email file.

        Args:
            email_path: Path to the email file

        Returns:
            Extracted email content
        """
        with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
            msg = email.message_from_file(f)

        subject = msg.get('Subject', '')
        sender = msg.get('From', '')

        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain' or content_type == 'text/html':
                    try:
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        body += str(part.get_payload())
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

        return f"From: {sender}\nSubject: {subject}\n\n{body}"

    def process_email(self, email_content: str) -> dict:
        """
        Determine if an email is spam.

        Args:
            email_content: The content of the email to check

        Returns:
            Dictionary with classification result and confidence
        """
        logger.info("Processing email for spam detection")

        # If we have a trained index, use it for retrieval
        if self.index:
            query_engine = self.index.as_query_engine(service_context=self.service_context)
            response = query_engine.query(
                f"Is this email spam? Analyze thoroughly: {email_content}"
            )
        else:
            # If no index, use direct prompting
            prompt = f"""
            Analyze the following email and determine if it is spam:

            {email_content}

            Provide a detailed analysis of why this is or is not spam.
            """
            response = self.llm.complete(prompt)

        # Use evaluator for final classification
        eval_result = self.evaluator.evaluate_response(
            query=email_content,
            response=Response(response)
        )

        is_spam = "spam" in eval_result.feedback.lower()
        confidence = 0.9 if "definitely" in eval_result.feedback.lower() else 0.7

        return {
            "is_spam": is_spam,
            "confidence": confidence,
            "analysis": response.response,
            "evaluator_feedback": eval_result.feedback
        }

    def process_email_file(self, email_path: str) -> dict:
        """
        Process an email file and determine if it's spam.

        Args:
            email_path: Path to the email file

        Returns:
            Dictionary with classification result and confidence
        """
        content = self._extract_email_content(email_path)
        return self.process_email(content)

    def batch_process(self, email_dir: str) -> pd.DataFrame:
        """
        Process all emails in a directory.

        Args:
            email_dir: Directory containing emails

        Returns:
            DataFrame with results
        """
        results = []

        for file_path in Path(email_dir).glob('*'):
            if file_path.is_file():
                try:
                    result = self.process_email_file(str(file_path))
                    results.append({
                        "file": file_path.name,
                        "is_spam": result["is_spam"],
                        "confidence": result["confidence"]
                    })
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = EmailSpamDetector()

    # Example email content
    example_email = """
    From: support@banking-secure.com
    Subject: URGENT: Your Account Has Been Compromised

    Dear Valued Customer,

    We have detected unusual activity on your account. Your account may have been compromised.

    To prevent further unauthorized transactions, please verify your identity immediately by clicking the link below:

    https://banking-secure-verify.com/login.php

    You will need to enter your full account details and PIN for verification purposes.

    Act now to protect your account!

    Security Team
    """

    # Process the example
    result = detector.process_email(example_email)

    print(f"Is spam: {result['is_spam']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Analysis: {result['analysis']}")