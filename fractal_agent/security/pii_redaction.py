"""
PII Redaction Module

Uses Microsoft Presidio for identifying and anonymizing Personally Identifiable Information (PII)
in text before logging or sending to LLMs.

Supported entity types:
- PERSON (names)
- EMAIL_ADDRESS
- PHONE_NUMBER
- US_SSN, US_DRIVER_LICENSE, US_PASSPORT
- CREDIT_CARD
- IBAN_CODE
- IP_ADDRESS
- LOCATION, US_ADDRESS
- DATE_TIME
- And more...

Author: BMad
Date: 2025-10-18
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PIIRedactor:
    """
    Redacts PII from text using Presidio.

    Usage:
        >>> redactor = PIIRedactor()
        >>> text = "John Doe's email is john@example.com"
        >>> redacted = redactor.redact(text)
        >>> print(redacted)
        <PERSON>'s email is <EMAIL_ADDRESS>
    """

    def __init__(self, language: str = 'en', score_threshold: float = 0.5):
        """
        Initialize PII redactor.

        Args:
            language: Language code (default: 'en')
            score_threshold: Confidence threshold for entity detection (0.0-1.0)
        """
        self.language = language
        self.score_threshold = score_threshold
        self._analyzer = None
        self._anonymizer = None

        # Try to import Presidio, but make it optional for testing
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            logger.info("Presidio PII redaction initialized")
        except ImportError:
            logger.warning(
                "Presidio not installed. PII redaction disabled. "
                "Install with: pip install presidio-analyzer presidio-anonymizer"
            )

    @property
    def is_available(self) -> bool:
        """Check if Presidio is available"""
        return self._analyzer is not None and self._anonymizer is not None

    def analyze(self, text: str, entities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Analyze text for PII without redacting.

        Args:
            text: Input text to analyze
            entities: Optional list of specific entity types to detect

        Returns:
            List of detected PII entities with scores
        """
        if not self.is_available:
            logger.warning("Presidio not available, returning empty results")
            return []

        results = self._analyzer.analyze(
            text=text,
            language=self.language,
            entities=entities,
            score_threshold=self.score_threshold
        )

        return [
            {
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "score": result.score,
                "text": text[result.start:result.end]
            }
            for result in results
        ]

    def redact(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        replacement: str = "<{entity_type}>"
    ) -> str:
        """
        Redact PII from text.

        Args:
            text: Input text to redact
            entities: Optional list of specific entity types to redact
            replacement: Replacement pattern (use {entity_type} for entity type)

        Returns:
            Text with PII redacted
        """
        if not self.is_available:
            logger.warning("Presidio not available, returning original text")
            return text

        # Analyze for PII
        analyzer_results = self._analyzer.analyze(
            text=text,
            language=self.language,
            entities=entities,
            score_threshold=self.score_threshold
        )

        # Anonymize
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results
        )

        return anonymized.text

    def redact_with_details(
        self,
        text: str,
        entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Redact PII and return both redacted text and details.

        Args:
            text: Input text to redact
            entities: Optional list of specific entity types to redact

        Returns:
            Dictionary with 'text' (redacted) and 'entities' (detected)
        """
        detected_entities = self.analyze(text, entities)
        redacted_text = self.redact(text, entities)

        return {
            "text": redacted_text,
            "entities_detected": len(detected_entities),
            "entities": detected_entities
        }


# Quick test/demo
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("PII Redaction Test")
    print("=" * 80)
    print()

    redactor = PIIRedactor()

    if not redactor.is_available:
        print("⚠️  Presidio not installed. Install with:")
        print("   pip install presidio-analyzer presidio-anonymizer")
        print()
        print("Note: This module will work but return original text unchanged.")
        exit(0)

    # Test cases
    test_cases = [
        "John Doe's email is john.doe@example.com",
        "Call me at 555-123-4567 or email jane@company.org",
        "My SSN is 123-45-6789",
        "The meeting is at 123 Main St, New York, NY 10001",
        "Credit card: 4111-1111-1111-1111"
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}:")
        print(f"  Original: {text}")

        result = redactor.redact_with_details(text)
        print(f"  Redacted: {result['text']}")
        print(f"  Detected: {result['entities_detected']} entities")

        if result['entities']:
            for entity in result['entities']:
                print(f"    - {entity['entity_type']}: {entity['text']} (score: {entity['score']:.2f})")
        print()

    print("=" * 80)
    print("✓ PII redaction test complete!")
    print("=" * 80)
