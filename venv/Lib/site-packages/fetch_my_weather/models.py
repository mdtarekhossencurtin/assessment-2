"""
Pydantic models for the wttr.in API responses.

These models define the structure of the JSON data returned by the wttr.in API,
providing type safety, validation, and easier access to weather data.
"""

from typing import Any

from pydantic import BaseModel, Field


class ResponseMetadata(BaseModel):
    """Metadata about the response from fetch-my-weather."""

    # Source of data
    is_real_data: bool = True  # Whether this is real data from the API
    is_cached: bool = False  # Whether this came from cache
    is_mock: bool = False  # Whether this is fallback mock data

    # Error information
    status_code: int | None = None  # HTTP status code if available
    error_type: str | None = None  # Type of error if any (e.g., "JSONDecodeError")
    error_message: str | None = None  # Detailed error message if any

    # Request information
    url: str | None = None  # URL that was requested
    timestamp: float | None = None  # When the request was made


class WeatherDesc(BaseModel):
    """Weather description model."""

    value: str


class WeatherIconUrl(BaseModel):
    """Weather icon URL model."""

    value: str


class Astronomy(BaseModel):
    """Astronomy information including sunrise, sunset, moonrise, moonset, etc."""

    moon_illumination: str | None = None
    moon_phase: str | None = None
    moonrise: str | None = None
    moonset: str | None = None
    sunrise: str | None = None
    sunset: str | None = None


class AreaName(BaseModel):
    """Area name model."""

    value: str


class Country(BaseModel):
    """Country model."""

    value: str


class Region(BaseModel):
    """Region model."""

    value: str


class HourlyForecast(BaseModel):
    """Hourly weather forecast data."""

    DewPointC: str | None = None
    DewPointF: str | None = None
    FeelsLikeC: str | None = None
    FeelsLikeF: str | None = None
    HeatIndexC: str | None = None
    HeatIndexF: str | None = None
    WindChillC: str | None = None
    WindChillF: str | None = None
    WindGustKmph: str | None = None
    WindGustMiles: str | None = None
    chanceoffog: str | None = None
    chanceoffrost: str | None = None
    chanceofhightemp: str | None = None
    chanceofovercast: str | None = None
    chanceofrain: str | None = None
    chanceofremdry: str | None = None
    chanceofsnow: str | None = None
    chanceofsunshine: str | None = None
    chanceofthunder: str | None = None
    chanceofwindy: str | None = None
    cloudcover: str | None = None
    humidity: str | None = None
    precipInches: str | None = None
    precipMM: str | None = None
    pressure: str | None = None
    pressureInches: str | None = None
    tempC: str | None = None
    tempF: str | None = None
    time: str | None = None
    uvIndex: str | None = None
    visibility: str | None = None
    visibilityMiles: str | None = None
    weatherCode: str | None = None
    weatherDesc: list[WeatherDesc] = Field(default_factory=list)
    weatherIconUrl: list[WeatherIconUrl] = Field(default_factory=list)
    winddir16Point: str | None = None
    winddirDegree: str | None = None
    windspeedKmph: str | None = None
    windspeedMiles: str | None = None


class CurrentCondition(BaseModel):
    """Current weather conditions."""

    FeelsLikeC: str | None = None
    FeelsLikeF: str | None = None
    cloudcover: str | None = None
    humidity: str | None = None
    localObsDateTime: str | None = None
    observation_time: str | None = None
    precipInches: str | None = None
    precipMM: str | None = None
    pressure: str | None = None
    pressureInches: str | None = None
    temp_C: str | None = None
    temp_F: str | None = None
    uvIndex: str | None = None
    visibility: str | None = None
    visibilityMiles: str | None = None
    weatherCode: str | None = None
    weatherDesc: list[WeatherDesc] = Field(default_factory=list)
    weatherIconUrl: list[WeatherIconUrl] = Field(default_factory=list)
    winddir16Point: str | None = None
    winddirDegree: str | None = None
    windspeedKmph: str | None = None
    windspeedMiles: str | None = None


class DailyForecast(BaseModel):
    """Daily weather forecast data."""

    astronomy: list[Astronomy] = Field(default_factory=list)
    avgtempC: str | None = None
    avgtempF: str | None = None
    date: str | None = None
    hourly: list[HourlyForecast] = Field(default_factory=list)
    maxtempC: str | None = None
    maxtempF: str | None = None
    mintempC: str | None = None
    mintempF: str | None = None
    sunHour: str | None = None
    totalSnow_cm: str | None = None
    uvIndex: str | None = None


class NearestArea(BaseModel):
    """Information about the nearest area."""

    areaName: list[AreaName] = Field(default_factory=list)
    country: list[Country] = Field(default_factory=list)
    latitude: str | None = None
    longitude: str | None = None
    population: str | None = None
    region: list[Region] = Field(default_factory=list)
    weatherUrl: list[WeatherIconUrl] = Field(default_factory=list)


class Request(BaseModel):
    """Information about the request that was made."""

    query: str | None = None
    type: str | None = None


class WeatherResponse(BaseModel):
    """Complete weather response from wttr.in API."""

    current_condition: list[CurrentCondition] = Field(default_factory=list)
    nearest_area: list[NearestArea] = Field(default_factory=list)
    request: list[Request] = Field(default_factory=list)
    weather: list[DailyForecast] = Field(default_factory=list)

    # Metadata for tracking response type and status
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class ResponseWrapper(BaseModel):
    """Wrapper for any response with metadata."""

    data: Any  # The actual response data (text, bytes, dict)
    metadata: ResponseMetadata  # Metadata about the response
