"""Copy to `typesafe_credentials.py` and fill in. NEVER commit the real file.

The key bills your TypeSafe account. Keep it out of version control (the real
file is in .gitignore) and out of any notebook you publish.

Get a key at https://console.typesafe.ai/keys.

    TYPESAFE_API_KEY = "sk-..."

Loaded by scripts/jev_client.py, which exports it to the environment BEFORE
`typesafe_sdk` is imported -- the SDK reads TYPESAFE_API_KEY from os.environ
(typesafe_sdk.constants.API_KEY_ENV). The key is never printed or logged.

RESEARCH TOOLING ONLY. Nothing under this key may be reachable from `main.py`.
The competition scores episodes in a sandbox with no network access and a
1.0s/turn actTimeout, so a hosted API call cannot be part of the submission;
`submit.py` pre-flight already hard-fails on a disallowed import in main.py.

Optional overrides -- leave empty to take the SDK defaults
(DEFAULT_BASE_URL='https://api.typesafe.ai', DEFAULT_MODEL='jev-latest').
Pin the versioned id rather than the alias once confidence thresholds are
tuned, because an alias moves when a new release ships.
"""

TYPESAFE_API_KEY = ""  # paste the key into typesafe_credentials.py, NOT here

# Pin a version (e.g. "jev-1.13.0") instead of the "jev-latest" alias once the
# audit's YES/NO thresholds have been calibrated against a specific model.
TYPESAFE_DEFAULT_MODEL = "jev-1.13.0"

TYPESAFE_BASE_URL = ""
