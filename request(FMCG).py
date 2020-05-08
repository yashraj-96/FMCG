import requests



url = 'http://localhost:5000/predict_api'

r = requests.post(url,json={'Product_Code':'4','Salesman_Code':'217','Month':'12','Target':'900'})


print(r.json())