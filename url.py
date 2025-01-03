import requests
import re
from urllib.parse import urlparse

# Your API Key for the Fact Check Tools API
API_KEY = "secrect key"

# Fact Check Tools API Endpoint
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Function to validate a URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Function to perform heuristic checks for malicious URLs
def is_malicious_url(url):
    # Simple heuristics: check for common phishing/malicious patterns
    suspicious_patterns = [
        r"\.tk$",          # Free domains often used in phishing
        r"login|secure",   # Suspicious keywords
        r"@|//.*@",        # Presence of "@" in the URL
        r"https?://\d+\.\d+\.\d+\.\d+",  # IP address in URL
        r"free|bonus|offer"  # Clickbait words
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False

# Function to search for fact-checked claims related to a URL
def search_fact_checked_claims(domain, query, language_code="en"):
    # Prepare the API parameters
    params = {
        "query": query,
        "languageCode": language_code,
        "key": API_KEY
    }
    
    # Make the GET request to the Fact Check Tools API
    response = requests.get(FACT_CHECK_API_URL, params=params)
    
    # Check the response status
    if response.status_code == 200:
        # Parse and return the claims
        results = response.json()
        return results.get("claims", [])
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to analyze a URL and provide a detailed response
def analyze_url(url):
    # Validate the URL
    if not is_valid_url(url):
        return "Invalid URL. Please enter a valid URL."
    
    # Parse the URL components
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    query = parsed_url.path + " " + (parsed_url.query or "")
    
    # Check if the URL is potentially malicious
    is_malicious = is_malicious_url(url)
    
    # Search for fact-checked claims related to the domain or query
    claims = search_fact_checked_claims(domain, query)
    
    # Generate a detailed response
    response = f"Analysis for URL: {url}\n"
    response += "-" * 50 + "\n"
    
    # Malicious URL analysis
    if is_malicious:
        response += "🚨 This URL exhibits suspicious patterns and might be malicious.\n"
    else:
        response += "✅ This URL does not exhibit obvious malicious patterns.\n"
    
    response += "\n"
    
    # Fact-checking results
    if claims:
        response += "🔍 Fact-Checked Claims Related to the URL:\n"
        for claim in claims[:3]:  # Limit to 3 claims for brevity
            response += f"  - Claim: {claim['text']}\n"
            response += f"    Claimed by: {claim.get('claimant', 'Unknown')}\n"
            response += f"    Rating: {claim['claimReview'][0]['textualRating']}\n"
            response += f"    Fact-checker: {claim['claimReview'][0]['publisher']['name']}\n"
            response += f"    More Info: {claim['claimReview'][0]['url']}\n\n"
    else:
        response += "No related fact-checked claims were found.\n"
    
    response += "-" * 50 + "\n"
    return response

# Test the function
if __name__ == "__main__":
    url_to_check = input("Enter a URL to analyze: ")
    result = analyze_url(url_to_check)
    print(result)

