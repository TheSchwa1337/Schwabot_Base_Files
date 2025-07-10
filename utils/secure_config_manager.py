import base64
import getpass
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet


class SecureConfigManager:
    """



    Secure API key management system for Schwabot.



    Integrates with existing mathematical framework for enhanced security.



    """

    def __init__(self, base_path: str = None):
        """Initialize the secure config manager with encrypted storage."""

        if base_path is None:

            base_path = os.path.join(os.path.dirname(__file__), "..", "secure")

        self.base_path = Path(base_path)

        self.secure_dir = self.base_path / "api_keys"

        self.config_file = self.base_path / "secure_config.json"

        self.key_file = self.base_path / "encryption.key"

        # Ensure secure directory exists

        self.secure_dir.mkdir(parents=True, exist_ok=True)

        # Initialize encryption

        self._initialize_encryption()

    def _initialize_encryption(self):
        """Initialize or load encryption key for secure storage."""

        if self.key_file.exists():

            with open(self.key_file, "rb") as f:

                self.encryption_key = f.read()

        else:

            # Generate new encryption key

            self.encryption_key = Fernet.generate_key()

            with open(self.key_file, "wb") as f:

                f.write(self.encryption_key)

        self.cipher = Fernet(self.encryption_key)

    def _hash_api_key(self, api_key: str, service_name: str) -> str:
        """Create a secure hash of the API key using Schwabot's mathematical framework."""'
        # Combine service name with key for unique hashing
        combined = f"{service_name}:{api_key}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return base64.b64encode()
            self.cipher.encrypt()
                data.encode("utf-8"))).decode("utf-8")

    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(base64.b64decode())
            encrypted_data.encode("utf-8"))).decode("utf-8")

    def secure_input(self, prompt: str, service_name: str) -> Dict[str, str]:
        """
        Securely input API key with masked display.
        Returns encrypted key and hash for storage.
        """
        print(f"\n🔐 Secure API Key Input for {service_name}")
        print("=" * 50)
        print(f"Enter your {service_name} API key (input will be, masked):")

        # Use getpass for secure input (masks the, input)
        api_key = getpass.getpass(prompt=f"{prompt}: ")

        if not api_key.strip():
            raise ValueError(f"API key for {service_name} cannot be empty")

        # Create hash for verification
        key_hash = self._hash_api_key(api_key, service_name)

        # Encrypt the actual key
        encrypted_key = self._encrypt_data(api_key)

        return {}
            "encrypted_key": encrypted_key,
            "key_hash": key_hash,
            "service": service_name,
            "timestamp": str(int(time.time())),
        }

    def store_api_key(self, service_name: str, prompt: str = None) -> bool:
        """
        Store an API key securely after user input.
        """
        if prompt is None:
            prompt = f"Enter {service_name} API key"

        try:
            key_data = self.secure_input(prompt, service_name)

            # Load existing config or create new
            config = self._load_config()
            config[service_name] = key_data

            # Save encrypted config
            self._save_config(config)

            print(f"✅ {service_name} API key stored securely")
            return True

        except Exception as e:
            print(f"❌ Failed to store {service_name} API key: {e}")
            return False

    def get_api_key(self, service_name: str) -> Optional[str]:
        """
        Retrieve and decrypt an API key for use.
        """
        try:
            config = self._load_config()

            if service_name not in config:
                return None

            key_data = config[service_name]
            encrypted_key = key_data["encrypted_key"]

            # Decrypt and return the key
            return self._decrypt_data(encrypted_key)

        except Exception as e:
            print(f"❌ Failed to retrieve {service_name} API key: {e}")
            return None

    def list_stored_services(self) -> list:
        """List all services with stored API keys."""
        config = self._load_config()
        return list(config.keys())

    def remove_api_key(self, service_name: str) -> bool:
        """Remove a stored API key."""
        try:
            config = self._load_config()

            if service_name in config:
                del config[service_name]
                self._save_config(config)
                print(f"✅ Removed {service_name} API key")
                return True

            return False

        except Exception as e:
            print(f"❌ Failed to remove {service_name} API key: {e}")
            return False

    def _load_config(self) -> Dict[str, Any]:
        """Load encrypted configuration file."""
        if not self.config_file.exists():
            return {}

        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_config(self, config: Dict[str, Any]):
        """Save encrypted configuration file."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

    def setup_required_keys(self) -> Dict[str, bool]:
        """
        Interactive setup for all required API keys.
        Returns status for each service.
        """
        required_services = {}
            "NEWS_API": "NewsAPI.org (for news, headlines)",
            "COINMARKETCAP_API": "CoinMarketCap API (for price, data)",
            "CCXT_API": "CCXT Exchange API",
            "COINBASE_API": "Coinbase API",
        }

        results = {}

        print("\n🔐 Schwabot Secure API Key Setup")
        print("=" * 50)
        print("This will securely store your API keys for Schwabot functionality.")
        print("Keys will be encrypted and hashed for security.\n")

        for service, description in required_services.items():
            print(f"\n📋 {description}")
            print(f"Service: {service}")

            # Check if already stored
            existing_key = self.get_api_key(service)
            if existing_key:
                print(f"✅ {service} key already stored")
                results[service] = True
                continue

            # Ask user if they want to set this key
            response = input(f"Set up {service} key? (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                success = self.store_api_key(service, f"Enter {service} API key")
                results[service] = success
            else:
                print(f"⏭️ Skipped {service}")
                results[service] = False

        return results


# Global instance for easy access
secure_config = SecureConfigManager()


def get_secure_api_key(service_name: str) -> Optional[str]:
    """Global function to retrieve API keys securely."""
    return secure_config.get_api_key(service_name)


def setup_api_keys() -> Dict[str, bool]:
    """Global function to setup all required API keys."""
    return secure_config.setup_required_keys()


if __name__ == "__main__":
    # Interactive setup when run directly
    print("🔐 Schwabot Secure API Key Manager")
    print("=" * 40)

    setup_results = setup_api_keys()

    print("\n📊 Setup Summary:")
    print("=" * 20)

    for service, success in setup_results.items():
        status = "✅ Ready" if success else "❌ Not configured"
        print(f"{service}: {status}")

    print(f"\n📋 Stored services: {secure_config.list_stored_services()}")
