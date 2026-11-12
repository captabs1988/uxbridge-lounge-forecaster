"""
index.py: the front door Vercel looks for.

Vercel automatically looks for a file called index.py with a variable called `app`.
Our real API lives in api/main.py, so this file just points Vercel to it.
"""
from api.main import app  # noqa: F401  (Vercel finds "app" here and serves it)
