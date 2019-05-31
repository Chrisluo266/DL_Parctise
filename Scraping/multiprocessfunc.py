import requests
import time
from bs4 import BeautifulSoup
import re

def crawl(url):
    response = requests.get(url)
    time.sleep(0.1)
    return response.text
def parse(base_url,html):
    soup = BeautifulSoup(html,'lxml')
    urls = soup.find_all('a',{"href":re.compile("^/.+?/$")})
    title = soup.find("h1").getText().strip()
    page_urls = set([base_url+url['href'] for url in urls])
    url = soup.find("meta",{'property':'og:url'})["content"]
    return title,page_urls,url