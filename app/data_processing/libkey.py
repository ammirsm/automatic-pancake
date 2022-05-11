import requests

headers = {
    "Accept": "application/vnd.api+json",
    "Accept-Language": "en-us",
    "Authorization": "Bearer 74138cd5-e5db-4ac2-9ee3-bad6764367e8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Origin": "https://libkey.io",
    "Pragma": "no-cache",
    "Referer": "https://libkey.io/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36",
    "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="101", "Google Chrome";v="101"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}

params = {
    "include": "issue,journal",
    "reload": "true",
}

doi = "10.1016/j.jclinepi.2021.03.013"
doi = doi.replace("/", "%2F")
url = "https://api.thirdiron.com/v2/articles/doi%3A" + doi

response = requests.get(url, params=params, headers=headers)
data = response.json()

print(data["data"]["attributes"]["openAccess"])
print(data["data"]["attributes"]["fullTextFile"])
