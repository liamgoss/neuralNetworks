import requests
dump_dict = {'reference_time': 1622876454,
             'sunset_time': 1622939025,
             'sunrise_time': 1622885142,
             'clouds': 1,
             'rain': {},
             'snow': {},
             'wind': {'speed': 3.09, 'deg': 240},
             'humidity': 94,
             'pressure': {'press': 1013, 'sea_level': None},
             'temperature': {'temp': 291.1, 'temp_kf': None, 'temp_max': 293.12, 'temp_min': 288.27, 'feels_like': 291.41},
             'status': 'Clear',
             'detailed_status': 'clear sky',
             'weather_code': 800,
             'weather_icon_name': '01n',
             'visibility_distance': 10000,
             'dewpoint': None,
             'humidex': None,
             'heat_index': None,
             'utc_offset': -14400,
             'uvi': None,
             'precipitation_probability': None}



#print(round((int(dump_dict['temperature']['temp']) - 273.15) * 9 / 5 + 32), "F")

res = requests.get("https://ipinfo.io")
print(res.json())
print(res.json()['city'])

info = {'ip': '67.181.250.65',
 'hostname': 'c-67-181-250-65.hsd1.ca.comcast.net',
        'city': 'Visalia',
        'region': 'California',
        'country': 'US',
        'loc': '36.3302,-119.2921',
        'org': 'AS33667 Comcast Cable Communications, LLC',
        'postal': '93278',
        'timezone': 'America/Los_Angeles',
        'readme': 'https://ipinfo.io/missingauth'}
