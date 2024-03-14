

import requests
import os

# Replace with your own Bing Search API subscription key
subscription_key = "JTeORjBATsUvARLv1VE4LMjF7o1jhzm4JXAiJaxMNTGMDhsu6rx2aQtmmW9p"

# Search query for Arnold Schwarzenegger
search_query = "Arnold Schwarzenegger"

# Number of images to download
num_images = 1000

# Output directory to save images
output_dir = "arnold_images"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Bing Image Search API endpoint
url = "https://api.bing.microsoft.com/v7.0/search"
#url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

# Set headers with API key
headers = {"Ocp-Apim-Subscription-Key": subscription_key}

# Parameters for the search query
params = {"q": search_query, "count": num_images}

# Make the API request
response = requests.get(url, headers=headers, params=params)
data = response.json()

print(data)
# Download and save the images
for i, image in enumerate(data["value"]):
    image_url = image["contentUrl"]
    image_filename = os.path.join(output_dir, f"arnold_{i+1}.jpg")
    try:
        image_data = requests.get(image_url).content
        with open(image_filename, "wb") as f:
            f.write(image_data)
        print(f"Downloaded image {i+1}/{num_images}: {image_filename}")
    except Exception as e:
        print(f"Error downloading image {i+1}: {str(e)}")

print(f"All {num_images} images downloaded and saved in {output_dir}.")

