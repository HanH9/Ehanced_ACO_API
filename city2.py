"""Implementation for the class representing City(longtitude,latitude)"""

import requests
from math import radians, sin, cos, sqrt, atan2


class City2:
    def __init__(self, name: str, longitude: float, latitude: float, demand: int):
        self.name = name
        self.longitude = longitude
        self.latitude = latitude
        self.demand = demand
        self.coordinate = (longitude, latitude)

    # def distance(self, other: "City2") -> float:
    #     """Calculates great circle distance to other city passed as other"""
    #     # Convert latitude and longitude from degrees to radians
    #     lat1, lon1 = radians(self.latitude), radians(self.longitude)
    #     lat2, lon2 = radians(other.latitude), radians(other.longitude)
    #
    #     # Haversine formula
    #     dlon = lon2 - lon1
    #     dlat = lat2 - lat1
    #     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    #     c = 2 * atan2(sqrt(a), sqrt(1 - a))
    #     radius_of_earth = 6371  # Earth radius in kilometers
    #     distance = radius_of_earth * c
    #     return distance

    def distance(self, other: "City2") -> float:
        """Calculates distance between two cities using Amap API"""
        # Replace "YOUR_API_KEY" with your actual Amap API key
        api_key = "6517ba72c7d078e1813aa5ac7545cede"
        url = f"https://restapi.amap.com/v3/direction/driving?key={api_key}&origin={self.longitude},{self.latitude}&destination={other.longitude},{other.latitude}"

        try:
            response = requests.get(url)
            data = response.json()
            if data["status"] == "1" and data["info"] == "OK":
                # Distance is returned in meters, convert to kilometers
                distance = float(data["route"]["paths"][0]["distance"]) / 1000
                # print(distance)
                return distance
            else:
                print("Error:", data["info"])
                return None
        except Exception as e:
            print("Error occurred while fetching data:", e)
            return None