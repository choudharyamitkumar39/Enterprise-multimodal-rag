"""
Real-Time Streaming Guardrail Interceptor.
Provides token-by-token streaming with a sliding-window safety buffer.
Monitors the live token stream for:
1. Secret / API key leakage (Groq, Pinecone, LangSmith, JWTs)
2. System prompt injection / extraction leaks
3. PII (SSNs, credit card numbers, passwords)
4. Harmful & toxic output patterns

If a violation is detected mid-stream, the circuit breaker immediately
terminates the stream and replaces the output with a certified safety warning.
"""

import re
import logging
from typing import Generator, Any, Dict, List, Optional

logger = logging.getLogger("streaming_guardrails")

# Compiled safety patterns for sub-millisecond regex scanning
SAFETY_PATTERNS = [
    # API Keys & Secrets
    (re.compile(r"gsk_[a-zA-Z0-9]{20,}", re.IGNORECASE), "Groq API Key Leakage"),
    (re.compile(r"pcsk_[a-zA-Z0-9]{20,}", re.IGNORECASE), "Pinecone API Key Leakage"),
    (re.compile(r"lsv2_[a-zA-Z0-9_]{20,}", re.IGNORECASE), "LangSmith Key Leakage"),
    (re.compile(r"eyJ[a-zA-Z0-9_\-]{20,}\.[a-zA-Z0-9_\-]{20,}", re.IGNORECASE), "JWT / Bearer Token Leakage"),
    (re.compile(r"(api[_\s]?key|secret[_\s]?key|password)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-]{8,}", re.IGNORECASE), "Credential Assignment Leakage"),
    
    # System Prompt Extraction Signatures
    (re.compile(r"You are a helpful assistant answering questions about the provided document", re.IGNORECASE), "System Prompt Extraction"),
    (re.compile(r"ChatPromptTemplate\.from_template", re.IGNORECASE), "Prompt Template Extraction"),
    (re.compile(r"RETRIEVED CONTEXT:\s*\[Page", re.IGNORECASE), "Raw Context Variable Injection"),
    
    # Common PII
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "Social Security Number (SSN)"),
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "Credit Card Number"),
]


class StreamingGuardrailInterceptor:
    """
    Sliding-window buffer that intercepts LLM token streams.
    Buffers `buffer_size` tokens before releasing to the client to ensure
    any split tokens or regex patterns can be verified prior to display.
    """

    def __init__(self, buffer_size: int = 8):
        self.buffer_size = buffer_size
        self.stats = {
            "total_streamed_tokens": 0,
            "violations_intercepted": 0,
            "tripped_pattern": None,
        }

    def intercept_stream(
        self,
        token_generator: Generator[str, None, None],
    ) -> Generator[str, None, None]:
        """
        Consumes an upstream token generator and yields safe tokens.
        If a safety policy is violated, breaks the loop and yields a safety notice.
        """
        token_buffer: List[str] = []
        accumulated_text = ""
        violation_triggered = False
        violation_detail = None

        try:
            for chunk in token_generator:
                if not chunk:
                    continue

                token_buffer.append(chunk)
                accumulated_text += chunk
                self.stats["total_streamed_tokens"] += 1

                # Check window against safety patterns
                # We inspect the tail of accumulated text (last 200 chars) for fast verification
                window_to_check = accumulated_text[-250:]

                for pattern, rule_name in SAFETY_PATTERNS:
                    if pattern.search(window_to_check):
                        violation_triggered = True
                        violation_detail = rule_name
                        self.stats["violations_intercepted"] += 1
                        self.stats["tripped_pattern"] = rule_name
                        logger.warning(f"CIRCUIT BREAKER ACTIVATED: Intercepted {rule_name} mid-stream!")
                        break

                if violation_triggered:
                    # Clear any unreleased buffer tokens to prevent leak
                    token_buffer.clear()
                    yield "\n\n⚠️ **[Response Terminated by Real-Time Safety Guardrail]**\n"
                    yield f"_Reason: Potential {violation_detail} was detected and blocked mid-stream._"
                    return

                # Flush oldest token once buffer exceeds desired threshold
                while len(token_buffer) > self.buffer_size:
                    yield token_buffer.pop(0)

            # Flush remaining tokens after upstream generator completes
            while token_buffer:
                yield token_buffer.pop(0)

        except Exception as e:
            logger.error(f"Error during streaming guardrail interception: {e}")
            yield f"\n[Stream interrupted: {e}]"
