(1) Create virtual env and activate
- "python3 -m venv venv"
- "source venv/bin/activate"

(2) Install dependencies
- "pip install -r requirements.txt"

(3) Start radar dsp client
- "python src/main -rp `<rada-port>`"
- add this flag --> "-sp `<speed-port>`" to connect to speed sensor

(4) Make sure "HOST" constant in main.py is set to the raspberry pi's host IP
- use "hostname -I" command (on pi) to retrieve 
