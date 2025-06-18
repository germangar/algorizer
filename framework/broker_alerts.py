import requests

def alert( message:str, url:str )->bool:
    headers = {
    'Content-Type': 'text/plain; charset=utf-8'
}
    req = requests.post(url, data=message.encode('utf-8'), headers=headers)
    return True

if __name__ == '__main__':
    url = 'https://webhook.site/ae09b310-eab0-4086-a0d1-2da80ab722d1'
    alert = 'This text'

    headers = {
        'Content-Type': 'text/plain; charset=utf-8'
    }

    req = requests.post(url, data=alert, headers=headers)

    # Optionally, you can print the response from the server
    print(req.status_code, req.text)