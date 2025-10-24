"""
fetch-my-weather - A beginner-friendly Python package for fetching weather data.

This package provides simple functions to fetch weather data and moon phases
from weather services, with built-in caching and error handling to make it suitable
for educational use.
"""

__version__ = "0.4.0"

from .core import (
    clear_cache,
    get_weather,
    set_cache_duration,
    set_mock_mode,
    set_user_agent,
)
from .models import (
    Astronomy,
    CurrentCondition,
    DailyForecast,
    HourlyForecast,
    NearestArea,
    ResponseMetadata,
    ResponseWrapper,
    WeatherResponse,
)

# For convenience, provide the most commonly used functions at the top level
__all__ = [
    # Functions
    "get_weather",
    "clear_cache",
    "set_cache_duration",
    "set_user_agent",
    "set_mock_mode",
    # Models
    "WeatherResponse",
    "CurrentCondition",
    "NearestArea",
    "DailyForecast",
    "HourlyForecast",
    "Astronomy",
    "ResponseMetadata",
    "ResponseWrapper",
]
