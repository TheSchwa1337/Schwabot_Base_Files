# -*- coding: utf-8 -*-
"""Dual Unicore Handler.

Provides cross-platform compatibility for Unicode character encoding and decoding,
essential for handling diverse glyphs and textual data within the Schwabot system.
It dynamically adapts to the operating system's default encoding, ensuring
seamless processing of character data across different environments.

Integrates with: [Other modules that handle text processing or glyphs]
"""

import sys
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class DualUnicoreHandler:
    """Handles cross-platform Unicode encoding and decoding.

    Dynamically adapts to the OS default encoding.
    """

    def __init__(self, encoding: Optional[str] = None):
        """Initialize the DualUnicoreHandler.

        Args:
            encoding: Optional, specific encoding to use. Defaults to system default.
        """
        self.encoding = encoding if encoding else sys.getdefaultencoding()
        self.platform = sys.platform
        logger.info(
            f"DualUnicoreHandler initialized for {self.platform} with encoding {self.encoding}"
        )

    def encode_text(self, text: str, target_encoding: Optional[str] = None) -> bytes:
        """Encode a string to bytes using the specified or default encoding.

        Args:
            text: The string to encode.
            target_encoding: Optional, encoding to use. Defaults to handler's encoding.

        Returns:
            Encoded bytes.
        """
        try:
            enc = target_encoding if target_encoding else self.encoding
            return text.encode(enc)
        except Exception as e:
            logger.error(f"Failed to encode text '{text}' with {enc}: {e}")
            raise

    def decode_bytes(self, data: bytes, target_encoding: Optional[str] = None) -> str:
        """Decode bytes to a string using the specified or default encoding.

        Args:
            data: The bytes to decode.
            target_encoding: Optional, encoding to use. Defaults to handler's encoding.

        Returns:
            Decoded string.
        """
        try:
            dec = target_encoding if target_encoding else self.encoding
            return data.decode(dec)
        except Exception as e:
            logger.error(f"Failed to decode bytes {data} with {dec}: {e}")
            raise

    def normalize_unicode(self, text: str, form: str = 'NFKC') -> str:
        """Normalize Unicode text to a specified form.

        Args:
            text: The string to normalize.
            form: Normalization form (NFC, NFD, NFKC, NFKD).

        Returns:
            Normalized string.
        """
        import unicodedata  # Lazy import to avoid unnecessary overhead
        try:
            return unicodedata.normalize(form, text)
        except Exception as e:
            logger.error(f"Failed to normalize Unicode text '{text}' with form {form}: {e}")
            raise

    def get_supported_encodings(self) -> List[str]:
        """Return a list of commonly supported encodings.

        Returns:
            List of encoding names.
        """
        return ['utf-8', 'latin-1', 'cp1252', 'ascii']


def main():
    """Demonstrates DualUnicoreHandler functionality."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    unicore_handler = DualUnicoreHandler()
    logger.info("DualUnicoreHandler module loaded successfully")

    # Test encoding and decoding
    test_string = "Hello, world! 👋 This is a test with some Unicode characters: éàü"
    print(f"\nOriginal String: {test_string}")

    encoded_bytes = unicore_handler.encode_text(test_string)
    print(f"Encoded Bytes ({unicore_handler.encoding}): {encoded_bytes}")

    decoded_string = unicore_handler.decode_bytes(encoded_bytes)
    print(f"Decoded String ({unicore_handler.encoding}): {decoded_string}")

    # Test with a different encoding
    try:
        encoded_latin1 = unicore_handler.encode_text("Résumé", "latin-1")
        print(f"Encoded (latin-1): {encoded_latin1}")
        decoded_latin1 = unicore_handler.decode_bytes(encoded_latin1, "latin-1")
        print(f"Decoded (latin-1): {decoded_latin1}")
    except Exception as e:
        logger.warning(f"Could not test latin-1 encoding: {e}")

    # Test Unicode normalization
    nfd_string = "Café"
    nfkc_string = unicore_handler.normalize_unicode(nfd_string, 'NFKC')
    print(f"\nOriginal (NFD): {nfd_string}, Normalized (NFKC): {nfkc_string}")

    # Test error handling
    try:
        unicore_handler.decode_bytes(b'\xed\xa0\x80', 'utf-8')  # Malformed UTF-8
    except Exception:
        logger.info("Caught expected decoding error.")


if __name__ == "__main__":
    main() 