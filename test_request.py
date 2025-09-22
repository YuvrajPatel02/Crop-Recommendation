import requests

url = "http://127.0.0.1:5000/api/recommend"
data = {
    "District": "Thrissur",
    "Season": "Kharif",
    "Rainfall": 250,
    "Temperature": 30,
    "LandType": "Irrigated",
    "Irrigation": "Canal",
    "SoilType": "Loamy"
}

res = requests.post(url, json=data)
print(res.json())
