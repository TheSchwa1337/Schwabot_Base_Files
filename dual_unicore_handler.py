#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Unicode Handler
====================

Handles dual unicode operations for Schwabot mathematical processing.
Provides unicode normalization and mathematical symbol handling.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DualUnicoreHandler:
    """Dual unicode handler for mathematical symbol processing."""

    def __init__(self):
        """Initialize dual unicode handler."""
        self.mathematical_symbols = {
            "psi": "Ψ",
            "phi": "Φ",
            "delta": "Δ",
            "epsilon": "ε",
            "lambda": "λ",
            "omega": "ω",
            "sigma": "σ",
            "tau": "τ",
            "pi": "π",
            "theta": "θ",
            "alpha": "α",
            "beta": "β",
            "gamma": "γ",
        }

        self.subscripts = {
            "0": "₀",
            "1": "₁",
            "2": "₂",
            "3": "₃",
            "4": "₄",
            "5": "₅",
            "6": "₆",
            "7": "₇",
            "8": "₈",
            "9": "₉",
            "n": "ₙ",
            "i": "ᵢ",
            "j": "ⱼ",
            "x": "ₓ",
            "y": "ᵧ",
        }

        self.superscripts = {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
            "n": "ⁿ",
            "i": "ⁱ",
            "+": "⁺",
            "-": "⁻",
        }

        logger.info("DualUnicoreHandler initialized")

    def normalize_text(self, text: str) -> str:
        """Normalize unicode text for mathematical processing."""
        try:
            # Normalize to NFC form
            normalized = unicodedata.normalize("NFC", text)
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize text: {e}")
            return text

    def get_mathematical_symbol(self, symbol_name: str) -> str:
        """Get mathematical symbol by name."""
        return self.mathematical_symbols.get(symbol_name.lower(), symbol_name)

    def add_subscript(self, text: str, subscript: str) -> str:
        """Add subscript to text."""
        try:
            subscript_chars = "".join(
                self.subscripts.get(char, char) for char in str(subscript)
            )
            return f"{text}{subscript_chars}"
        except Exception as e:
            logger.error(f"Failed to add subscript: {e}")
            return f"{text}_{subscript}"

    def add_superscript(self, text: str, superscript: str) -> str:
        """Add superscript to text."""
        try:
            superscript_chars = "".join(
                self.superscripts.get(char, char) for char in str(superscript)
            )
            return f"{text}{superscript_chars}"
        except Exception as e:
            logger.error(f"Failed to add superscript: {e}")
            return f"{text}^{superscript}"

    def format_mathematical_expression(self, expression: str) -> str:
        """Format mathematical expression with proper unicode symbols."""
        try:
            # Replace common mathematical symbols
            formatted = expression
            for name, symbol in self.mathematical_symbols.items():
                formatted = formatted.replace(f"{name}_", symbol)
                formatted = formatted.replace(name.upper(), symbol)

            return formatted
        except Exception as e:
            logger.error(f"Failed to format expression: {e}")
            return expression

    def encode_dual_unicode(self, primary: str, secondary: str) -> str:
        """Encode dual unicode representation."""
        try:
            primary_normalized = self.normalize_text(primary)
            secondary_normalized = self.normalize_text(secondary)
            return f"{primary_normalized}⊕{secondary_normalized}"
        except Exception as e:
            logger.error(f"Failed to encode dual unicode: {e}")
            return f"{primary}+{secondary}"

    def decode_dual_unicode(self, encoded: str) -> tuple[str, str]:
        """Decode dual unicode representation."""
        try:
            if "⊕" in encoded:
                parts = encoded.split("⊕", 1)
                return parts[0], parts[1] if len(parts) > 1 else ""
            elif "+" in encoded:
                parts = encoded.split("+", 1)
                return parts[0], parts[1] if len(parts) > 1 else ""
            else:
                return encoded, ""
        except Exception as e:
            logger.error(f"Failed to decode dual unicode: {e}")
            return encoded, ""

    def process_mathematical_text(self, text: str) -> str:
        """Process mathematical text with unicode handling."""
        try:
            # Normalize the text
            processed = self.normalize_text(text)

            # Format mathematical expressions
            processed = self.format_mathematical_expression(processed)

            return processed
        except Exception as e:
            logger.error(f"Failed to process mathematical text: {e}")
            return text

    def validate_unicode_integrity(self, text: str) -> bool:
        """Validate unicode text integrity."""
        try:
            # Check if text can be encoded/decoded properly
            encoded = text.encode("utf-8")
            decoded = encoded.decode("utf-8")
            return decoded == text
        except Exception as e:
            logger.error(f"Unicode integrity validation failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get handler status information."""
        return {
            "mathematical_symbols_count": len(self.mathematical_symbols),
            "subscripts_count": len(self.subscripts),
            "superscripts_count": len(self.superscripts),
            "handler_active": True,
        }
