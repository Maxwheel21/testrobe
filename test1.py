import time
import hmac
import hashlib
import requests
import json

def generate_totp(secret, time_step=30, digits=10):
    # Calculate the current time step
    current_time = int(time.time())
    time_counter = current_time // time_step

    # Encode the time counter to a byte array
    time_counter_bytes = time_counter.to_bytes(8, 'big')

    # Create an HMAC-SHA-512 hash
    hmac_hash = hmac.new(secret.encode(), time_counter_bytes, hashlib.sha512).digest()

    # Extract the dynamic binary code (DBC)
    offset = hmac_hash[-1] & 0x0F
    binary_code = ((hmac_hash[offset] & 0x7F) << 24 |
                   (hmac_hash[offset + 1] & 0xFF) << 16 |
                   (hmac_hash[offset + 2] & 0xFF) << 8 |
                   (hmac_hash[offset + 3] & 0xFF))

    # Calculate the TOTP value
    totp = binary_code % (10 ** digits)

    # Pad the TOTP with leading zeros if necessary
    return str(totp).zfill(digits)

# User information
email = "lordherobrine98@gmail.com"
secret = email + "HENNGECHALLENGE003"
github_url = "https://gist.github.com/Maxwheel21/eed9c2c10119244087810fdfc13fedee"
language = "python"

# Generate TOTP password
password = generate_totp(secret)

# Construct JSON payload
payload = {
    "github_url": github_url,
    "contact_email": email,
    "solution_language": language
}

# Encode JSON payload
json_payload = json.dumps(payload)

# Prepare HTTP headers
headers = {
    "Content-Type": "application/json"
}

# Prepare Basic Auth header
auth = (email, password)

# URL for the POST request
url = "https://api.challenge.hennge.com/challenges/003"

# Send POST request
response = requests.post(url, headers=headers, data=json_payload, auth=auth)

# Print response status and body
print("Response Status Code:", response.status_code)
print("Response Body:", response.text)
