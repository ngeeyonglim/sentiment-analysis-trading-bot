import requests
from bs4 import BeautifulSoup
import csv

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}

# the URL of the home page of the target website
base_url = 'https://quotes.toscrape.com'

quotes = []

# Function to extract quotes from a page
def extract_quotes_from_page(soup):
    quote_elements = soup.find_all('div', class_='quote')
    
    for quote_element in quote_elements:
        # extract the text of the quote
        text = quote_element.find('span', class_='text').text
        # extract the author of the quote
        author = quote_element.find('small', class_='author').text

        # extract the tag <a> HTML elements related to the quote
        tag_elements = quote_element.select('.tags .tag')

        # store the list of tag strings in a list
        tags = []
        for tag_element in tag_elements:
            tags.append(tag_element.text)

        quotes.append({
            'text': text,
            'author': author,
            'tags': ', '.join(tags)
        })

# Get the first page
page = requests.get(base_url, headers=headers)
soup = BeautifulSoup(page.text, 'html.parser')

# Extract quotes from the first page
extract_quotes_from_page(soup)

# Get the "Next →" HTML element
next_li_element = soup.find('li', class_='next')

# If there is a next page to scrape
while next_li_element is not None:
    next_page_relative_url = next_li_element.find('a', href=True)['href']
    
    # Get the new page
    page = requests.get(base_url + next_page_relative_url, headers=headers)
    
    # Parse the new page
    soup = BeautifulSoup(page.text, 'html.parser')
    
    # Apply the same scraping logic to the new page
    extract_quotes_from_page(soup)
    
    # Look for the "Next →" HTML element in the new page
    next_li_element = soup.find('li', class_='next')

# At this point, the 'quotes' list contains all quotes from all pages
print(f"Total quotes scraped: {len(quotes)}")

with open('quotes.csv', 'w', encoding='utf-8', newline='') as csv_file:
	# initializing the writer object to insert data
	# in the CSV file
	writer = csv.writer(csv_file)

	# writing the header of the CSV file
	writer.writerow(['Text', 'Author', 'Tags'])

	# writing each row of the CSV
	for quote in quotes:
	    writer.writerow(quote.values())

# terminating the operation and releasing the resources
csv_file.close()