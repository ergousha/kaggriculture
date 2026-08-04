"""Copy to `kaggle_credentials.py` and fill in. NEVER commit the real file.

Anyone who obtains kaggle_credentials.py has full API access to your Kaggle
account — it can submit, download and delete on your behalf. Keep it out of
version control (it is in .gitignore) and out of any notebook you publish.

Get credentials at https://www.kaggle.com/settings/api.

Two auth methods work with kaggle 2.2.4; set EITHER.

1. Access token (what the Kaggle docs now recommend). "Generate New Token"
   under the API section gives you a token string:

       KAGGLE_API_TOKEN = "xxxxxxxxxxxxxxxx"

2. Legacy username + API key, as found in a downloaded kaggle.json:

       KAGGLE_USERNAME = "your_username"
       KAGGLE_KEY = "0123456789abcdef0123456789abcdef"

If both are set the access token wins, matching the Kaggle client's own
precedence (KaggleApi.authenticate tries the access token first, then the
legacy API key).
"""

KAGGLE_API_TOKEN = ""

KAGGLE_USERNAME = "your_username"
KAGGLE_KEY = "your_api_key"
