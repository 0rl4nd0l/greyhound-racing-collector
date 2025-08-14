#!/usr/bin/env python3
"""
OpenAI Connectivity Verification System
=======================================

Implements OpenAI API connectivity verification with:
1. Environment variable validation
2. Minimal chat completion for authentication testing
3. Retry/rate-limit handling with exponential back-off
4. Mock client fallback for local development

This module ensures proper OpenAI integration for the Greyhound Analysis Predictor.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockOpenAIClient:
    """Mock OpenAI client for local development when API key is not available"""

    def __init__(self):
        self.chat = MockChat()

    class MockChat:
        def __init__(self):
            self.completions = MockCompletions()

        class MockCompletions:
            def create(self, **kwargs):
                """Mock chat completion response"""
                return MockResponse()

    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
            self.usage = MockUsage()

        class MockChoice:
            def __init__(self):
                self.message = MockMessage()

            class MockMessage:
                def __init__(self):
                    self.content = json.dumps(
                        {
                            "status": "mock_response",
                            "message": "This is a mock response for local development. Real OpenAI API not available.",
                            "timestamp": datetime.now().isoformat(),
                            "confidence": 0.1,
                        }
                    )

        class MockUsage:
            def __init__(self):
                self.total_tokens = 10
                self.prompt_tokens = 5
                self.completion_tokens = 5


class OpenAIConnectivityVerifier:
    """Handles OpenAI API connectivity verification and management"""

    def __init__(self):
        self.api_key = None
        self.client = None
        self.is_mock = False
        self.connection_verified = False

    def load_api_key(self) -> bool:
        """
        Load OPENAI_API_KEY from environment variables.
        Returns True if key is found, False otherwise.
        """
        try:
            from dotenv import load_dotenv

            load_dotenv()  # Load .env file if available
        except ImportError:
            logger.info(
                "python-dotenv not available, using system environment variables"
            )

        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logger.error("❌ OPENAI_API_KEY not found in environment variables.")
            logger.error("Please set the OPENAI_API_KEY in your .env file:")
            logger.error("OPENAI_API_KEY=sk-your-actual-api-key-here")
            return False

        if (
            self.api_key.startswith("sk-test")
            or self.api_key == "your_openai_api_key_here"
        ):
            logger.warning("⚠️  Test or placeholder API key detected.")
            logger.warning("Using mock client for local development.")
            return False

        logger.info("✅ OPENAI_API_KEY found in environment")
        return True

    def initialize_client(self, use_mock: bool = False) -> bool:
        """
        Initialize OpenAI client or mock client.
        Returns True if initialization successful.
        """
        if use_mock or not self.api_key:
            logger.info("🔧 Initializing mock OpenAI client for local development")
            self.client = MockOpenAIClient()
            self.is_mock = True
            return True

        try:
            import openai

            logger.info(
                f"Initializing OpenAI client with API key: {self.api_key[:10]}..."
            )
            logger.info(f"OpenAI version: {openai.__version__}")
            # Initialize with minimal parameters to avoid any conflicts
            self.client = openai.OpenAI(api_key=self.api_key)
            self.is_mock = False
            logger.info("✅ OpenAI client initialized successfully")
            return True
        except ImportError:
            logger.error(
                "❌ OpenAI Python package not installed. Install with: pip install openai"
            )
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            return False

    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify OpenAI connection with minimal chat completion.
        Returns verification result with status and details.
        """
        if not self.client:
            return {
                "success": False,
                "error": "OpenAI client not initialized",
                "timestamp": datetime.now().isoformat(),
            }

        if self.is_mock:
            logger.info("🔧 Using mock client - connection verification skipped")
            return {
                "success": True,
                "mock_mode": True,
                "message": "Mock client active for local development",
                "timestamp": datetime.now().isoformat(),
            }

        logger.info("🔍 Verifying OpenAI API connection...")

        try:
            # Minimal test completion
            response = self._make_api_call_with_retry(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "Hello, this is a connection test. Please respond with 'Connected'.",
                    },
                ],
                max_tokens=10,
                temperature=0.1,
            )

            if response:
                content = response.choices[0].message.content.strip()
                tokens_used = response.usage.total_tokens

                self.connection_verified = True
                logger.info("✅ OpenAI API connection verified successfully")

                from config.openai_config import get_openai_config
                cfg = get_openai_config()
                return {
                    "success": True,
                    "response_content": content,
                    "tokens_used": tokens_used,
                    "model": cfg.model,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {
                    "success": False,
                    "error": "No response received from OpenAI API",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"❌ OpenAI API connection verification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _make_api_call_with_retry(
        self, messages, max_tokens=100, temperature=0.3, max_retries=5
    ):
        """
        Make OpenAI API call with exponential backoff retry logic.
        Handles rate limiting and transient errors.
        """
        import openai
        from config.openai_config import get_openai_config
        cfg = get_openai_config()

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response

            except openai.RateLimitError as e:
                wait_time = (2**attempt) + (
                    attempt * 0.1
                )  # Exponential backoff with jitter
                logger.warning(
                    f"⏳ Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

            except openai.APITimeoutError as e:
                wait_time = 2**attempt
                logger.warning(
                    f"⏳ API timeout (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s..."
                )
                time.sleep(wait_time)

            except openai.APIConnectionError as e:
                wait_time = 2**attempt
                logger.warning(
                    f"🔌 Connection error (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s..."
                )
                time.sleep(wait_time)

            except openai.APIError as e:
                if "quota" in str(e).lower():
                    logger.error(
                        "💳 API quota exceeded. Please check your OpenAI billing."
                    )
                    raise
                wait_time = 2**attempt
                logger.warning(
                    f"⚠️  API error (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time}s..."
                )
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"❌ Unexpected error during API call: {e}")
                raise

        logger.error(f"❌ Max retries ({max_retries}) exceeded for OpenAI API call")
        raise RuntimeError(f"Max retries ({max_retries}) exceeded for OpenAI API call")

    def get_enhanced_client(self):
        """
        Get client wrapper with built-in retry logic.
        Returns a client that automatically handles retries.
        """
        if self.is_mock:
            return self.client

        return EnhancedOpenAIClient(self.client, self._make_api_call_with_retry)

    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report of OpenAI connectivity"""
        return {
            "api_key_loaded": bool(self.api_key),
            "api_key_valid": bool(
                self.api_key and not self.api_key.startswith("sk-test")
            ),
            "client_initialized": bool(self.client),
            "is_mock_mode": self.is_mock,
            "connection_verified": self.connection_verified,
            "timestamp": datetime.now().isoformat(),
        }


class EnhancedOpenAIClient:
    """Wrapper for OpenAI client with built-in retry logic"""

    def __init__(self, client, retry_function):
        self.client = client
        self._make_api_call_with_retry = retry_function
        self.chat = self.EnhancedChat(retry_function)

    class EnhancedChat:
        def __init__(self, retry_function):
            self._make_api_call_with_retry = retry_function
            self.completions = EnhancedOpenAIClient.EnhancedCompletions(retry_function)

    class EnhancedCompletions:
        def __init__(self, retry_function):
            self._make_api_call_with_retry = retry_function

        def create(self, **kwargs):
            """Create chat completion with automatic retry logic"""
            messages = kwargs.get("messages", [])
            max_tokens = kwargs.get("max_tokens", 100)
            temperature = kwargs.get("temperature", 0.3)

            return self._make_api_call_with_retry(
                messages=messages, max_tokens=max_tokens, temperature=temperature
            )


def main():
    """Test the OpenAI connectivity verification system"""
    print("🚀 Testing OpenAI Connectivity Verification System")
    print("=" * 60)

    verifier = OpenAIConnectivityVerifier()

    # Step 1: Load API key
    print("\n1️⃣  Loading API key from environment...")
    key_loaded = verifier.load_api_key()

    # Step 2: Initialize client
    print("\n2️⃣  Initializing OpenAI client...")
    client_init = verifier.initialize_client(use_mock=not key_loaded)

    if not client_init:
        print("❌ Failed to initialize client")
        return

    # Step 3: Verify connection
    print("\n3️⃣  Verifying API connection...")
    verification_result = verifier.verify_connection()

    print(f"\n📊 Verification Result:")
    print(json.dumps(verification_result, indent=2))

    # Step 4: Status report
    print("\n4️⃣  Status Report:")
    status = verifier.get_status_report()
    print(json.dumps(status, indent=2))

    # Step 5: Test enhanced client
    if verification_result.get("success"):
        print("\n5️⃣  Testing enhanced client with retry logic...")
        enhanced_client = verifier.get_enhanced_client()

        try:
            test_response = enhanced_client.chat.completions.create(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                max_tokens=20,
                temperature=0.1,
            )
            print(
                f"✅ Enhanced client test successful: {test_response.choices[0].message.content}"
            )
        except Exception as e:
            print(f"❌ Enhanced client test failed: {e}")

    print("\n🏁 OpenAI connectivity verification complete!")


if __name__ == "__main__":
    main()
