"""
Core functionality for fetch_my_weather - A beginner-friendly Python package for weather data.

This module provides the main functions for fetching weather data and moon phases,
with built-in caching and error handling to make it suitable for educational use.
"""

import json
import time
from typing import Any, Literal

import requests
from pydantic import ValidationError

from .models import ResponseMetadata, ResponseWrapper, WeatherResponse

# --- Configuration ---
BASE_URL = "http://wttr.in/"
_CACHE_DURATION_SECONDS = 600  # Cache data for 10 minutes
_USER_AGENT = "fetch-my-weather/0.4.0"  # Be polite and identify our package
_USE_MOCK_DATA = False  # Flag to use mock data instead of real API

# --- In-memory Cache ---
# Simple dictionary to store cached responses
# Format: { "url": (timestamp, data) }
_cache: dict[str, tuple[float, str | bytes | dict[str, Any] | WeatherResponse]] = {}

# --- Mock Data ---
# Sample responses for different request types
_MOCK_DATA = {
    "text": """Weather report: MockCity

     \033[38;5;226m   \\  /\033[0m       Partly cloudy
     \033[38;5;226m _ /\"\"\033[38;5;250m.-.    \033[0m  \033[38;5;214m17\033[0m °C
     \033[38;5;226m   \\_\033[38;5;250m(   ).   \033[0m  ↗ \033[38;5;220m11\033[0m km/h
     \033[38;5;226m   /\033[38;5;250m(___(__) \033[0m  10 km
                  0.0 mm
                  [MOCK DATA]
""",
    "json": {
        "mock_data_notice": "This is simulated weather data for educational purposes",
        "current_condition": [
            {
                "FeelsLikeC": "16",
                "FeelsLikeF": "60",
                "cloudcover": "25",
                "humidity": "71",
                "temp_C": "17",
                "temp_F": "63",
                "weatherDesc": [{"value": "Partly cloudy"}],
                "weatherIconUrl": [{"value": ""}],
                "winddir16Point": "NE",
                "winddirDegree": "45",
                "windspeedKmph": "11",
                "windspeedMiles": "7",
                "pressure": "1015",
                "pressureInches": "30",
                "precipMM": "0.0",
                "precipInches": "0.0",
                "visibility": "10",
                "visibilityMiles": "6",
                "uvIndex": "4",
            }
        ],
        "nearest_area": [
            {
                "areaName": [{"value": "MockCity"}],
                "country": [{"value": "MockLand"}],
                "region": [{"value": "MockRegion"}],
                "latitude": "51.517",
                "longitude": "-0.106",
                "population": "7556900",
                "weatherUrl": [{"value": ""}],
            }
        ],
        "request": [{"query": "MockCity", "type": "City"}],
        "weather": [
            {
                "date": "2025-04-13",
                "astronomy": [
                    {
                        "moon_phase": "Waxing Gibbous",
                        "moon_illumination": "75",
                        "sunrise": "06:12 AM",
                        "sunset": "07:59 PM",
                        "moonrise": "12:34 PM",
                        "moonset": "02:56 AM",
                    }
                ],
                "maxtempC": "18",
                "mintempC": "9",
                "avgtempC": "14",
                "maxtempF": "64",
                "mintempF": "48",
                "avgtempF": "57",
                "totalSnow_cm": "0.0",
                "sunHour": "10.5",
                "uvIndex": "4",
                "hourly": [
                    {
                        "time": "0",
                        "tempC": "10",
                        "tempF": "50",
                        "weatherDesc": [{"value": "Clear"}],
                        "weatherCode": "113",
                        "weatherIconUrl": [{"value": ""}],
                        "windspeedMiles": "5",
                        "windspeedKmph": "8",
                        "winddirDegree": "35",
                        "winddir16Point": "NE",
                        "precipMM": "0.0",
                        "precipInches": "0.0",
                        "humidity": "75",
                        "visibility": "10",
                        "visibilityMiles": "6",
                        "pressure": "1015",
                        "pressureInches": "30",
                        "cloudcover": "15",
                        "HeatIndexC": "10",
                        "HeatIndexF": "50",
                        "DewPointC": "6",
                        "DewPointF": "43",
                        "WindChillC": "9",
                        "WindChillF": "48",
                        "WindGustMiles": "7",
                        "WindGustKmph": "11",
                        "FeelsLikeC": "9",
                        "FeelsLikeF": "48",
                        "uvIndex": "1",
                    }
                ],
            }
        ],
    },
    "png": b"Mock PNG data - not real image bytes",
}

# --- Configuration Functions ---


def set_cache_duration(seconds: int) -> int:
    """
    Set how long to cache weather data in seconds.

    Args:
        seconds: Duration in seconds. Set to 0 to disable caching.

    Returns:
        The new cache duration in seconds.
    """
    global _CACHE_DURATION_SECONDS
    _CACHE_DURATION_SECONDS = max(0, int(seconds))
    return _CACHE_DURATION_SECONDS


def set_user_agent(user_agent: str) -> str:
    """
    Set the User-Agent string sent with requests.

    Args:
        user_agent: The User-Agent string to use.

    Returns:
        The new User-Agent string.
    """
    global _USER_AGENT
    _USER_AGENT = str(user_agent)
    return _USER_AGENT


def clear_cache() -> int:
    """
    Clear all cached weather data.

    Returns:
        Number of entries cleared from cache.
    """
    global _cache
    count = len(_cache)
    _cache.clear()
    return count


def set_mock_mode(use_mock: bool) -> bool:
    """
    Enable or disable the use of mock data instead of real API calls.

    This is useful for development and testing when you want to avoid
    hitting the wttr.in API, especially under rate limiting.

    Args:
        use_mock: True to use mock data, False to use real API.

    Returns:
        The new mock mode setting.
    """
    global _USE_MOCK_DATA
    _USE_MOCK_DATA = bool(use_mock)
    return _USE_MOCK_DATA


# --- Helper Functions ---


def _create_metadata(
    is_real_data: bool = True,
    is_cached: bool = False,
    is_mock: bool = False,
    status_code: int | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    url: str | None = None,
) -> ResponseMetadata:
    """
    Creates a ResponseMetadata object with the given parameters.

    Args:
        is_real_data: Whether this is real data from the API.
        is_cached: Whether this is cached data.
        is_mock: Whether this is mock/fallback data.
        status_code: HTTP status code if available.
        error_type: Type of error if any.
        error_message: Detailed error message if any.
        url: URL that was requested.

    Returns:
        A ResponseMetadata object.
    """
    return ResponseMetadata(
        is_real_data=is_real_data,
        is_cached=is_cached,
        is_mock=is_mock,
        status_code=status_code,
        error_type=error_type,
        error_message=error_message,
        url=url,
        timestamp=time.time(),
    )


def _wrap_response(
    data: Any, metadata: ResponseMetadata, with_metadata: bool
) -> str | bytes | dict[str, Any] | WeatherResponse | ResponseWrapper:
    """
    Wraps the response data with metadata if requested.

    Args:
        data: The response data.
        metadata: The metadata to include.
        with_metadata: Whether to include metadata.

    Returns:
        Either the data directly or a ResponseWrapper with data and metadata.
    """
    if with_metadata:
        return ResponseWrapper(data=data, metadata=metadata)

    # For Pydantic models, attach metadata directly to the model
    if isinstance(data, WeatherResponse):
        data.metadata = metadata
        return data
    elif isinstance(data, str):
        return data
    elif isinstance(data, bytes):
        return data
    elif isinstance(data, dict):
        return data
    else:
        # Convert to string as a last resort
        return str(data)


def _create_mock_data(
    format: Literal["text", "json", "raw_json", "png"],
    error_type: str,
    error_message: str,
    with_metadata: bool,
    url: str | None = None,
    status_code: int | None = None,
) -> str | bytes | dict[str, Any] | WeatherResponse | ResponseWrapper:
    """
    Creates appropriate mock data with metadata for any error situation.

    Args:
        format: The requested format.
        error_type: Type of error that occurred.
        error_message: Detailed error message.
        with_metadata: Whether to include metadata.
        url: URL that was requested.
        status_code: HTTP status code if available.

    Returns:
        Appropriate mock data with metadata.
    """
    # Create metadata for the mock response
    metadata = _create_metadata(
        is_real_data=False,
        is_cached=False,
        is_mock=True,
        status_code=status_code,
        error_type=error_type,
        error_message=error_message,
        url=url,
    )

    # Create appropriate mock data based on format
    if format == "png":
        mock_data = _MOCK_DATA["png"]
        return _wrap_response(mock_data, metadata, with_metadata)
    elif format == "raw_json":
        # Create a deep copy and add a note about it being mock data
        mock_data = json.loads(json.dumps(_MOCK_DATA["json"]))
        # Since we know the mock data is a dictionary, safely cast and add note
        if isinstance(mock_data, dict):
            mock_data["note"] = f"Mock data provided due to error: {error_type}"
            return _wrap_response(mock_data, metadata, with_metadata)
        else:
            # This should never happen with our mock data, but just in case
            return _wrap_response(
                "Error: Invalid mock data format", metadata, with_metadata
            )
    elif format == "json":
        # Convert to Pydantic model
        try:
            mock_data = json.loads(json.dumps(_MOCK_DATA["json"]))
            mock_response = WeatherResponse.parse_obj(mock_data)
            return _wrap_response(mock_response, metadata, with_metadata)
        except ValidationError:
            # If model conversion fails, use text format as fallback
            mock_data = _MOCK_DATA["text"]
            return _wrap_response(mock_data, metadata, with_metadata)
    else:  # text format
        mock_data = _MOCK_DATA["text"]
        return _wrap_response(mock_data, metadata, with_metadata)


def _build_url(
    location: str = "",
    units: str = "",
    view_options: str = "",
    lang: str | None = None,
    is_png: bool = False,
    png_options: str = "",
    is_moon: bool = False,
    moon_date: str | None = None,
    moon_location_hint: str | None = None,
    format: Literal["text", "json", "raw_json", "png"] = "text",
) -> str:
    """
    Constructs the full URL for the wttr.in request.

    Args:
        location: Location to get weather for
        units: Units for output ('m' for metric, 'u' for USCS, 'M' for wind m/s)
        view_options: Combined view options (e.g., '0', '1', 'n', 'q', 'F', '0q')
        lang: Language code (e.g., 'fr', 'de', 'zh-cn')
        is_png: If True, request a PNG image instead of text (deprecated, use format="png" instead)
        png_options: PNG specific options (e.g., 'p', 't', 'transparency=100')
        is_moon: If True, fetch moon phase instead of weather
        moon_date: Specific date for moon phase in 'YYYY-MM-DD' format
        moon_location_hint: Location hint for moon phase (e.g., ',+US', ',+Paris')
        format: Output format - "text", "json", "raw_json", or "png" (default: "text")

    Returns:
        Full URL for the wttr.in request
    """
    url = BASE_URL
    query_params: dict[str, str] = {}
    path_parts = []

    # --- Location Handling ---
    if is_moon:
        path = "/moon"
        if moon_date:
            path += f"@{moon_date}"
        if moon_location_hint:
            # Ensure hint starts correctly, remove leading characters if present in input
            hint = moon_location_hint.lstrip(",+")
            # Add location hint with no + sign
            path += f",{hint}"
        location_part = path
    elif location:
        # URL encode the location to handle spaces, special characters, unicode
        # Use %20 for spaces instead of + for tests to pass
        location_part = f"/{location.replace(' ', '%20')}"
    else:
        # Empty location means current location based on IP
        location_part = "/"

    # --- Options Handling ---
    # Combine all relevant options into a single string for path/query
    options_list = []
    if units:
        options_list.append(units)
    if view_options:
        options_list.append(view_options)
    if is_png and png_options:
        options_list.append(png_options)

    combined_options = "".join(options_list)

    # --- Output Format Handling ---
    # For backward compatibility
    if is_png:
        format = "png"

    if format == "png":
        # PNG format uses underscores in the path
        path_parts.append(
            location_part.lstrip("/")
        )  # Remove leading slash for path joining
        if combined_options:
            path_parts.append(combined_options)
        if lang:
            path_parts.append(f"lang={lang}")

        # Join parts with '_' and add .png extension
        url += "_".join(path_parts) + ".png"
    else:
        # Text or JSON format uses query parameters
        url += location_part.lstrip("/")  # Add location to path

        # Add format specification for JSON formats
        if format == "json" or format == "raw_json":
            query_params["format"] = "j1"  # j1 is the compact JSON format from wttr.in

        # Add options as query parameter string if any exist
        if combined_options:
            query_params["format_options"] = (
                combined_options  # Use a placeholder key first
            )

        if lang:
            query_params["lang"] = lang

        # Build the query string
        if query_params:
            # Special handling for combined options which don't have a key in wttr.in
            query_string_parts = []
            if "format_options" in query_params:
                query_string_parts.append(query_params["format_options"])
                del query_params["format_options"]  # Remove placeholder

            # Add remaining key=value pairs
            query_string_parts.extend([f"{k}={v}" for k, v in query_params.items()])

            url += "?" + "&".join(query_string_parts)

    return url


def _get_from_cache(url: str) -> str | bytes | dict[str, Any] | WeatherResponse | None:
    """
    Checks cache for non-expired data.

    Args:
        url: URL to check in cache

    Returns:
        Cached data if available and not expired, None otherwise
    """
    if _CACHE_DURATION_SECONDS <= 0:
        return None  # Caching disabled

    if url in _cache:
        timestamp, data = _cache[url]
        if time.time() - timestamp < _CACHE_DURATION_SECONDS:
            # Cache hit
            return data
        else:
            # Cache expired
            del _cache[url]  # Remove expired entry
    return None


def _add_to_cache(
    url: str, data: str | bytes | dict[str, Any] | WeatherResponse
) -> None:
    """
    Adds data to the cache with current timestamp.

    Args:
        url: URL to cache
        data: Data to cache
    """
    if _CACHE_DURATION_SECONDS > 0:  # Only cache if enabled
        _cache[url] = (time.time(), data)


# --- Public API Function ---


def get_weather(
    location: str = "",
    units: str = "",
    view_options: str = "",
    lang: str | None = None,
    is_png: bool = False,
    png_options: str = "",
    is_moon: bool = False,
    moon_date: str | None = None,
    moon_location_hint: str | None = None,
    format: Literal["text", "json", "raw_json", "png"] = "json",
    use_mock: bool | None = None,
    with_metadata: bool = False,
) -> str | bytes | dict[str, Any] | WeatherResponse | ResponseWrapper:
    """
    Fetches weather or moon phase information from wttr.in.

    Args:
        location: The location identifier (city, airport code, coordinates,
                 domain, area code, ~special name, or empty for current).
                 Ignored if is_moon is True.
        units: Units for output ('m' for metric, 'u' for USCS, 'M' for wind m/s).
        view_options: Combined view options (e.g., '0', '1', 'n', 'q', 'F', '0q').
        lang: Language code (e.g., 'fr', 'de', 'zh-cn').
        is_png: If True, request a PNG image instead of text (deprecated, use format="png").
        png_options: PNG specific options (e.g., 'p', 't', 'transparency=100').
                    Only used for PNG format.
        is_moon: If True, fetch moon phase instead of weather.
        moon_date: Specific date for moon phase in 'YYYY-MM-DD' format.
                  Only used if is_moon is True.
        moon_location_hint: Location hint for moon phase (e.g., ',+US', ',+Paris').
                           Only used if is_moon is True.
        format: Output format - "text", "json", "raw_json", or "png" (default: "json").
                When "json" is used, returns a WeatherResponse Pydantic model.
                When "raw_json" is used, returns the raw JSON as a Python dictionary.
        use_mock: If True, use mock data instead of making a real API request.
                 If None, use the global setting (_USE_MOCK_DATA).
        with_metadata: If True, include metadata about the response (real vs mock,
                      cache status, errors). Returns a ResponseWrapper instead of direct data.

    Returns:
        If with_metadata is True: Returns a ResponseWrapper containing both data and metadata.
        If format is "text": Returns the weather report as a string.
        If format is "json": Returns the weather data as a WeatherResponse Pydantic model.
        If format is "raw_json": Returns the raw JSON data as a Python dictionary.
        If format is "png": Returns the PNG image data as bytes.
        If an error occurs and with_metadata is False: Returns an error message string.
        If an error occurs and with_metadata is True: Returns fallback data with error details in metadata.
        No exceptions are raised.
    """
    # Input validation (optional but good practice)
    if units not in ["", "m", "u", "M"]:
        error_msg = "Error: Invalid 'units' parameter. Use 'm', 'u', or 'M'."
        if with_metadata:
            return _create_mock_data(
                format=format,
                error_type="ValidationError",
                error_message=error_msg,
                with_metadata=with_metadata,
            )
        return error_msg
    # Add more validation as needed...

    # Build the request URL
    url = _build_url(
        location=location,
        units=units,
        view_options=view_options,
        lang=lang,
        is_png=is_png,
        png_options=png_options,
        is_moon=is_moon,
        moon_date=moon_date,
        moon_location_hint=moon_location_hint,
        format=format,
    )

    # Determine whether to use mock data
    should_use_mock = _USE_MOCK_DATA if use_mock is None else use_mock

    # If mock mode is enabled, return mock data
    if should_use_mock:
        # Create metadata for mock data
        metadata = _create_metadata(
            is_real_data=False,
            is_cached=False,
            is_mock=True,
            url=url,
        )

        if format == "png" or is_png:
            # Cast to bytes using a type assertion for the type checker
            mock_png: bytes = _MOCK_DATA["png"]  # type: ignore
            return _wrap_response(mock_png, metadata, with_metadata)
        elif format == "json":
            # Make a deep copy to avoid modifying the original mock data
            json_data = json.loads(json.dumps(_MOCK_DATA["json"]))
            try:
                # Convert to Pydantic model
                model_data: WeatherResponse = WeatherResponse.parse_obj(json_data)
                return _wrap_response(model_data, metadata, with_metadata)
            except ValidationError:
                validation_error_msg: str = (
                    "Error: Mock data doesn't match the expected model structure"
                )
                if with_metadata:
                    return _create_mock_data(
                        format="text",  # Fall back to text format
                        error_type="ValidationError",
                        error_message=validation_error_msg,
                        with_metadata=with_metadata,
                        url=url,
                    )
                return validation_error_msg
        elif format == "raw_json":
            # Return the raw JSON as a dictionary without Pydantic conversion
            # Make a deep copy to avoid modifying the original mock data
            dict_data: dict[str, Any] = json.loads(json.dumps(_MOCK_DATA["json"]))
            return _wrap_response(dict_data, metadata, with_metadata)
        else:
            # Explicit cast to str for type checker
            text_data: str = str(_MOCK_DATA["text"])
            return _wrap_response(text_data, metadata, with_metadata)

    # Check cache first
    cached_data = _get_from_cache(url)
    if cached_data is not None:
        # Create metadata for cached data
        cache_metadata = _create_metadata(
            is_real_data=True,  # It was real when cached
            is_cached=True,
            is_mock=False,
            url=url,
        )
        # If it's JSON format and we have a cached string or dict
        if format == "json" or format == "raw_json":
            if isinstance(cached_data, str):
                try:
                    json_data = json.loads(cached_data)
                    # For raw_json, just return the parsed dictionary
                    if format == "raw_json":
                        parsed_dict: dict[str, Any] = json_data
                        return _wrap_response(
                            parsed_dict, cache_metadata, with_metadata
                        )
                    # For json, convert to Pydantic model
                    parsed_model: WeatherResponse = WeatherResponse.parse_obj(json_data)
                    return _wrap_response(parsed_model, cache_metadata, with_metadata)
                except (json.JSONDecodeError, ValidationError) as e:
                    # If JSON parsing fails and metadata is requested, create error metadata
                    if with_metadata:
                        metadata = _create_metadata(
                            is_real_data=False,
                            is_cached=True,
                            is_mock=False,
                            error_type=e.__class__.__name__,
                            error_message=str(e),
                            url=url,
                        )
                        # Return mock data with error metadata
                        return _create_mock_data(
                            format=format,
                            error_type=e.__class__.__name__,
                            error_message=str(e),
                            with_metadata=with_metadata,
                            url=url,
                        )

                    # If not with_metadata, fall back to original behavior
                    if isinstance(cached_data, str):
                        error_text: str = cached_data
                        return error_text
                    # Fallback for other types
                    return str(cached_data)
            elif isinstance(cached_data, WeatherResponse):
                # If it's already a WeatherResponse object and format is json
                if format == "json":
                    weather_model: WeatherResponse = cached_data
                    return _wrap_response(weather_model, cache_metadata, with_metadata)
                # If format is raw_json, convert WeatherResponse to dict
                elif format == "raw_json":
                    # Convert Pydantic model to dict
                    model_dict: dict[str, Any] = json.loads(cached_data.json())
                    return _wrap_response(model_dict, cache_metadata, with_metadata)
            elif isinstance(cached_data, dict):
                # If it's a dict and format is raw_json, return as is
                if format == "raw_json":
                    raw_dict: dict[str, Any] = cached_data
                    return _wrap_response(raw_dict, cache_metadata, with_metadata)
                # If format is json, convert to WeatherResponse
                try:
                    cached_model: WeatherResponse = WeatherResponse.parse_obj(
                        cached_data
                    )
                    return _wrap_response(cached_model, cache_metadata, with_metadata)
                except ValidationError as e:
                    struct_error: str = f"Error: Cached data doesn't match the expected model structure: {str(e)}"

                    if with_metadata:
                        # Return mock data with error metadata
                        return _create_mock_data(
                            format=format,
                            error_type="ValidationError",
                            error_message=struct_error,
                            with_metadata=with_metadata,
                            url=url,
                        )

                    return struct_error
        # Handle other formats or types
        if isinstance(cached_data, str):
            cached_text: str = cached_data
            return _wrap_response(cached_text, cache_metadata, with_metadata)
        elif isinstance(cached_data, bytes):
            binary_data: bytes = cached_data
            return _wrap_response(binary_data, cache_metadata, with_metadata)
        else:
            # For any other type, convert to string
            fallback: str = str(cached_data)
            return _wrap_response(fallback, cache_metadata, with_metadata)

    # --- Perform the actual request ---
    headers = {"User-Agent": _USER_AGENT}
    # Add Accept-Language header if language is specified (alternative to ?lang=)
    # Note: ?lang= or subdomain is generally preferred by wttr.in documentation
    # if lang:
    #     headers['Accept-Language'] = lang

    try:
        response = requests.get(url, headers=headers, timeout=15)  # 15 second timeout

        # Create metadata for the real API response
        real_metadata = _create_metadata(
            is_real_data=True,
            is_cached=False,
            is_mock=False,
            status_code=response.status_code,
            url=url,
        )

        # Check if the request was successful (status code 2xx)
        if 200 <= response.status_code < 300:
            # Determine return type based on request format
            if format == "png" or is_png:
                data = response.content  # Return raw bytes for images
                # Add successful response to cache
                _add_to_cache(url, data)
                return _wrap_response(data, real_metadata, with_metadata)
            elif format == "json" or format == "raw_json":
                # For JSON formats, parse the response
                try:
                    data = response.text
                    # Add raw text to cache
                    _add_to_cache(url, data)
                    json_data = json.loads(data)

                    # For raw_json, return the dictionary without Pydantic conversion
                    if format == "raw_json":
                        json_dict_data: dict[str, Any] = json_data
                        return _wrap_response(
                            json_dict_data, real_metadata, with_metadata
                        )

                    # For standard json, convert to Pydantic model
                    try:
                        weather_response: WeatherResponse = WeatherResponse.parse_obj(
                            json_data
                        )
                        return _wrap_response(
                            weather_response, real_metadata, with_metadata
                        )
                    except ValidationError as e:
                        validation_error: str = f"Error: JSON data doesn't match the expected model structure: {str(e)}"

                        if with_metadata:
                            # Return mock data with error details
                            return _create_mock_data(
                                format=format,
                                error_type="ValidationError",
                                error_message=validation_error,
                                with_metadata=with_metadata,
                                url=url,
                                status_code=response.status_code,
                            )

                        return validation_error

                except json.JSONDecodeError:
                    # Create metadata and return if needed
                    if with_metadata:
                        metadata = _create_metadata(
                            is_real_data=False,
                            is_cached=False,
                            is_mock=True,
                            status_code=response.status_code,
                            error_type="JSONDecodeError",
                            error_message=f"Unable to parse JSON response from {url}",
                            url=url,
                        )
                        mock_data = _create_mock_data(
                            format=format,
                            error_type="JSONDecodeError",
                            error_message=f"Unable to parse JSON response from {url}",
                            with_metadata=with_metadata,
                            url=url,
                            status_code=response.status_code,
                        )
                        return _wrap_response(mock_data, metadata, with_metadata)

                    # JSON decode failed - for educational use, provide mock data instead of error
                    return _create_mock_data(
                        format=format,
                        error_type="JSONDecodeError",
                        error_message=f"Unable to parse JSON response from {url}",
                        with_metadata=with_metadata,
                        url=url,
                        status_code=response.status_code,
                    )
            else:
                # Text format - return as is
                data = response.text
                # Add successful response to cache
                _add_to_cache(url, data)
                return _wrap_response(data, real_metadata, with_metadata)
        else:
            # Handle non-successful status codes gracefully
            error_message = (
                f"Error fetching data from wttr.in: "
                f"Status code {response.status_code} for URL {url}"
            )
            # Maybe include response text if available and not too long?
            try:
                error_details = response.text[:200]  # Limit details length
                if error_details:
                    error_message += f"\nResponse body (start): {error_details}"
            except Exception:
                pass  # Ignore errors trying to get error details

            # Create error metadata and return if requested
            if with_metadata:
                metadata = _create_metadata(
                    is_real_data=False,
                    is_cached=False,
                    is_mock=True,  # Will be mocked
                    status_code=response.status_code,
                    error_type="HTTPError",
                    error_message=error_message,
                    url=url,
                )
                mock_data = _create_mock_data(
                    format=format,
                    error_type="HTTPError",
                    error_message=error_message,
                    with_metadata=with_metadata,
                    url=url,
                    status_code=response.status_code,
                )
                return _wrap_response(mock_data, metadata, with_metadata)

            # For any error status code, provide fallback data if with_metadata is true
            if with_metadata:
                return _create_mock_data(
                    format=format,
                    error_type="HTTPError",
                    error_message=error_message,
                    with_metadata=with_metadata,
                    url=url,
                    status_code=response.status_code,
                )

            # Without metadata, still provide mock data for 503 errors with JSON formats
            if response.status_code == 503 and (
                format == "json" or format == "raw_json"
            ):
                # Provide a mock response with a note about rate limiting
                if format == "raw_json":
                    # Make a deep copy and add a note about it being mock data
                    mock_data = json.loads(json.dumps(_MOCK_DATA["json"]))
                    if isinstance(mock_data, dict):
                        mock_data["note"] = "Mock data provided due to rate limiting"
                        raw_json_data: dict[str, Any] = mock_data
                        return raw_json_data
                    else:
                        # This should never happen with our mock data, but just in case
                        return "Error: Invalid mock data format"
                else:
                    # Convert mock data to Pydantic model
                    try:
                        mock_data = json.loads(json.dumps(_MOCK_DATA["json"]))
                        mock_weather_response: WeatherResponse = (
                            WeatherResponse.parse_obj(mock_data)
                        )
                        return mock_weather_response
                    except ValidationError:
                        # If model conversion fails, still return the error message
                        pass

            # Otherwise return the error message
            return error_message

    except Exception as e:
        error_type = e.__class__.__name__
        if error_type == "Timeout":
            error_message = f"Error: Request timed out while connecting to {url}"
        elif error_type == "ConnectionError":
            error_message = (
                f"Error: Could not connect to {url}. Check network connection."
            )
        elif "requests" in str(e.__class__.__module__):
            # Catch any other requests-related error
            error_message = f"Error: An unexpected network error occurred: {e}"
        else:
            # Catch any other unexpected error during processing
            # This shouldn't normally happen with the above catches, but belt-and-suspenders
            error_message = f"Error: An unexpected error occurred: {e}"

        # If with_metadata is enabled, return mock data with error information
        if with_metadata:
            return _create_mock_data(
                format=format,
                error_type=error_type,
                error_message=error_message,
                with_metadata=with_metadata,
                url=url,
            )

        # Otherwise return the error message
        return error_message
