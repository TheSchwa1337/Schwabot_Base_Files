from price_bridge import get_secure_price
from secure_config_manager import get_secure_api_key

    Any,
    Dict,
    List,
    Optional,
    asyncio,
    from,
    get_secure_api_key,
    get_secure_price,
    hashlib,
    import,
    requests,
    rt,
    typing,
    utils.secure_config_manager,
)


except ImportError:

    # Fallback for direct execution


def pull_news_headlines(): -> List[str]:
    """



    Pulls news headlines related to a given query using the NewsAPI.



    Uses secure config manager for API key retrieval.



    """

    api_key = get_secure_api_key("NEWS_API")

    if not api_key:

        print(" NEWS_API key not found. Please run the secure config setup first.")

        print("   Run: python utils/secure_config_manager.py")

        return []

    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}"

    try:

        response = requests.get(url, timeout=5)

        response.raise_for_status()

        articles = response.json().get("articles", [])

        return [article["title"]]
                for article in articles][:5]  # Get top 5 headlines

    except requests.exceptions.RequestException as e:

        print(f" Error fetching news headlines: {e}")

        return []


def get_btc_price(): -> float:
    """



    Fetches the current Bitcoin price in USD using the secure price bridge.



    This is a synchronous wrapper for the async price bridge.



    """

    try:

        # Run async function in sync context

        loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)

        price_data = loop.run_until_complete(get_secure_price("BTC"))

        loop.close()

        if price_data:

            return price_data.price

        else:

            print(" Failed to get BTC price from secure price bridge")

            return 0.0

    except Exception as e:

        print(f" Error getting BTC price: {e}")

        return 0.0


def get_secure_price_data(): -> Optional[Dict[str, Any]]:
    """



    Get price data using the secure price bridge with Schwabot's mathematical framework.'



    """

    try:

        # Run async function in sync context

        loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)

        price_data = loop.run_until_complete(get_secure_price(symbol))

        loop.close()

        if price_data:

            return price_data.to_dict()

        else:

            print(f" Failed to get price data for {symbol}")

            return None

    except Exception as e:

        print(f" Error getting price data: {e}")

        return None


def hash_market_state(): -> str:
    """



    Generates a SHA-256 hash based on news headlines and Bitcoin price.



    This hash serves as a unique identifier for a specific market state.



    """

    text_to_hash = " ".join(sorted(news_titles)) + str(price)

    return hashlib.sha256(text_to_hash.encode("utf-8")).hexdigest()


def create_market_snapshot(): -> Optional[Dict[str, Any]]:
    """



    Create a complete market snapshot with news, price, and hash.



    Returns None if critical data is missing.



    """

    print(" Creating market snapshot...")

    # Fetch news headlines

    news = pull_news_headlines()

    if not news:

        print(" Failed to fetch news headlines")

        return None

    # Fetch price data using secure price bridge

    price_data = get_secure_price_data()

    if not price_data:

        print(" Failed to fetch price data")

        return None

    # Generate market state hash

    market_hash = hash_market_state(news, price_data["price"])

    snapshot = {}
        "timestamp": price_data["timestamp"],
        "news_headlines": news,
        "price_data": price_data,
        "market_hash": market_hash,
        "status": "success",
    }

    return snapshot


def display_market_snapshot(snapshot: Dict[str, Any]):
    """Display a formatted market snapshot."""

    if not snapshot:

        print(" No market snapshot available")

        return

    print("\n" + "=" * 60)

    print(" SCHWABOT MARKET STATE SNAPSHOT")

    print("=" * 60)

    # Display news headlines

    print("\n Top News Headlines:")

    for i, title in enumerate(snapshot["news_headlines"], 1):

        print(f"  {i}. {title}")

    # Display price data

    price_data = snapshot["price_data"]

    print("\n Price Data:")

    print(f"  Symbol: {price_data['symbol']}")

    print(f"  Price: ${price_data['price']:,.2f} {price_data['currency']}")

    print(f"  Source: {price_data['source']}")

    print(f"  Timestamp: {price_data['timestamp']}")

    # Display mathematical framework data

    if price_data.get("price_hash"):

        print(f"  Price Hash: {price_data['price_hash'][:16]}...")

    if price_data.get("market_state_hash"):

        print()
            f"  Market State Hash: {price_data['market_state_hash'][:16]}...")

    # Display market hash

    print("\n Market State Hash:")

    print(f"  {snapshot['market_hash']}")

    print("\n" + "=" * 60)


async def create_async_market_snapshot(): -> Optional[Dict[str, Any]]:
    """



    Async version of market snapshot creation for better performance.



    """

    print(" Creating async market snapshot...")

    # Fetch news headlines

    news = pull_news_headlines()

    if not news:

        print(" Failed to fetch news headlines")

        return None

    # Fetch price data using async price bridge

    price_data = await get_secure_price("BTC")

    if not price_data:

        print(" Failed to fetch price data")

        return None

    # Generate market state hash

    market_hash = hash_market_state(news, price_data.price)

    snapshot = {}
        "timestamp": price_data.timestamp,
        "news_headlines": news,
        "price_data": price_data.to_dict(),
        "market_hash": market_hash,
        "status": "success",
    }

    return snapshot


if __name__ == "__main__":

    # Test the market data functionality

    print(" Testing Schwabot Market Data Integration")

    print("=" * 50)

    # Check if API keys are configured

    news_key = get_secure_api_key("NEWS_API")

    if not news_key:

        print(" NEWS_API key not configured.")

        print("   Please run: python utils/secure_config_manager.py")

        exit(1)

    print(" NEWS_API key found")

    # Create and display market snapshot

    snapshot = create_market_snapshot()

    if snapshot:

        display_market_snapshot(snapshot)

        print("\n Market data integration working correctly!")

    else:

        print("\n Failed to create market snapshot")

        print("   Check your API keys and internet connection")
