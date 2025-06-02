import requests

url = 'https://webhook.site/8cf90478-6c35-4eb4-ac61-e72417ac5a3b'
alert = 'This text'

headers = {
    'Content-Type': 'text/plain; charset=utf-8'
}

req = requests.post(url, data=alert, headers=headers)

# Optionally, you can print the response from the server
print(req.status_code, req.text)
