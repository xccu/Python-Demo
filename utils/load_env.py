#pip install python-dotenv
from dotenv import load_dotenv, find_dotenv

# from dotenv import load_dotenv, find_dotenv
# dev= find_dotenv()
# _ = load_dotenv(dev) # read local .env file

def load():
    dev= find_dotenv()
    return load_dotenv(dev) # read local .env file