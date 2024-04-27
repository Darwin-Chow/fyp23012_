import cloudscraper
from bs4 import BeautifulSoup as soup
import pandas as pd
import requests


scraper = cloudscraper.create_scraper()

html = scraper.get("https://etherscan.io/address-analytics?m=light&a=0x00062d1dd1afb6fb02540ddad9cdebfe568e0d89&lg=en&cc=USD&__cf_chl_rt_tk=m7n3bX5B5jPkWXKHuWJxTacTrAYY848fZ4H6BGl8AC4-1698804209-0-gaNycGzNCyU")

page_soup = soup(html.text, "html.parser")

highestEth = page_soup.find(id='high_bal_eth_value')

print("testing:\n", highestEth)