from __future__ import annotations

import httpx
from anthropic import beta_async_tool

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


@beta_async_tool
async def get_weather(city: str) -> str:
    """Get the current live weather for a city.

    Args:
        city: City name, e.g. "San Francisco"
    """
    # @beta_async_tool reads this docstring + the type hints above to build
    # the tool's JSON schema automatically -- no hand-written schema dict.
    async with httpx.AsyncClient(timeout=10) as client:
        geo = (await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
        )).json()
        results = geo.get("results")
        if not results:
            return f"Could not find a location matching '{city}'."

        loc = results[0]
        label = f"{loc['name']}, {loc.get('admin1', '')} {loc.get('country', '')}".strip()

        current = (await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "temperature_unit": "fahrenheit",
            },
        )).json()["current"]

    condition = WEATHER_CODES.get(current["weather_code"], f"code {current['weather_code']}")
    return (
        f"Live weather for {label} as of {current['time']}: "
        f"{current['temperature_2m']}F, {condition}, wind {current['wind_speed_10m']} mph."
    )
