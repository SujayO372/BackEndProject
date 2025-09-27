# Quick Start for — Backend (Python)

Minimal instructions to create a virtual environment, run the backend, and test the HTTP endpoints.
Replace the dummy IP 192.0.2.1 below with your real server IP or hostname.

Prerequisites

Ubuntu / Debian-like Linux (commands shown use apt)

Python 3.10 available in apt repo

requirements.txt present in project root

Backend.py present in project root

1. Create & activate a Python virtual environment
# install venv support (run as root or with sudo)
sudo apt update
sudo apt install -y python3.10-venv

# create the venv (runs Python 3.10)
python3.10 -m venv venv

# activate the virtual environment (bash / sh)
source venv/bin/activate

# you should now see (venv) in your prompt

2. Install Python dependencies
# Install packages listed in requirements.txt into the activated venv
pip install --upgrade pip
pip install -r requirements.txt


If you encounter SSL / wheel / compiler issues, make sure build essentials are installed:

sudo apt install -y build-essential libssl-dev libffi-dev python3-dev

3. Run the backend
# from project root (while venv is activated)
python ./Backend.py


The app should start and bind to the address/port configured in your code (commonly 0.0.0.0:5000 or 127.0.0.1:5000).

Leave the terminal open while the server runs. Use Ctrl+C to stop.

4. Health-check endpoint (GET)

Use curl (or open in browser) to verify the service is up.

# replace 192.0.2.1 with your server IP or localhost if running locally
curl http://192.0.2.1:5000/health


Expected: a JSON or text response indicating service status (e.g., {"status":"ok"} or healthy), depending on your implementation.

5. Send a POST query (JSON body)

This shows how to POST the query string "hello how are you" to the /query endpoint.

curl -X POST http://192.0.2.1:5000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "hello how are you"}'


If your backend expects form data instead of JSON, use:

curl -X POST http://192.0.2.1:5000/query \
     -d "query=hello how are you"

6. Common troubleshooting

ModuleNotFoundError: ensure venv is activated and dependencies are installed (run pip show <package> to verify).

Port already in use: another process is listening on port 5000; either stop it or change your app’s port.

No response from curl: check firewall rules and that the app binds to 0.0.0.0 (if you need external access) rather than 127.0.0.1.

Permissions: if you used sudo to run Python/venv files, prefer not to — create venv and run as a normal user to avoid file-permission issues.

7. Stop & cleanup
# Stop app: Ctrl+C in the running terminal

# Deactivate virtualenv
deactivate

# To remove the virtualenv (cleanup)
rm -rf venv

Notes

The IP 192.0.2.1 in this document is a placeholder (TEST-NET-1). Replace it with your real IP or localhost / 127.0.0.1 when testing locally.

Commit this README.md to GitHub for easy onboarding of other devs.
