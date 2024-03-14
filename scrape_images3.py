import requests
from bs4 import BeautifulSoup
import os

def download_arnold_images(search_term, num_images, output_dir):
  """
  Searches for images using Google Images (for educational purposes only)
  and creates a directory to store the downloaded images.

  Args:
      search_term (str): The term to search for (e.g., "Arnold Schwarzenegger").
      num_images (int): The number of images to download (limited to 100 for demonstration).
      output_dir (str): The directory to store the downloaded images.
  """

  # Disclaimer: Downloading content without explicit permission can be illegal.
  # This code is provided for educational purposes only and should not be used
  # to violate copyright laws.

  os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

  user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"
  headers = {'User-Agent': user_agent}

  for i in range(num_images):
    # Construct search URL (replace with a suitable image search engine)
    search_url = f"https://www.google.com/search?q={search_term}&start={i*10}"

    try:
      response = requests.get(search_url, headers=headers)
      response.raise_for_status()  # Raise an exception for failed requests

      soup = BeautifulSoup(response.content, 'html.parser')

      # Extract image URLs (replace with a more robust image scraping method)
      # **Important:** This approach is for demonstration purposes only and might not
      # be reliable for all websites. It's crucial to respect copyright laws
      # and avoid unauthorized scraping.
      img_links = [img.get('src') for img in soup.find_all('img')]

      print(img_links)
      
      for j, img_link in enumerate(img_links):
        if img_link:
          filename = f"{output_dir}/{search_term}_{i}_{j}.jpg"

          try:
            img_response = requests.get(img_link, stream=True)
            img_response.raise_for_status()

            with open(filename, 'wb') as f:
              for chunk in img_response.iter_content(1024):
                f.write(chunk)

            print(f"Downloaded image {j+1} from URL: {img_link}")
          except requests.exceptions.RequestException as e:
            print(f"Error downloading image {j+1}: {e}")

    except requests.exceptions.RequestException as e:
      print(f"Error fetching search results: {e}")

if __name__ == "__main__":
  search_term = "Arnold Schwarzenegger"
  num_images = 100  # Limited for demonstration (consider copyright implications)
  output_dir = "arnold_images"

  download_arnold_images(search_term, num_images, output_dir)
