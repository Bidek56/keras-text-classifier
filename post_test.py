import requests

text = "Good Quality Dog Food,I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most."

res = requests.post('http://localhost:5000/predict', data={"text":text} )
if res.ok:
    print (res.json())