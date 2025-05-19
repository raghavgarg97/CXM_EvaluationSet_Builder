import json

def parse_json(s, default_json = None):
    try:
        first = s.find('{')
        last = s.rfind('}')
        if first == -1 or last == -1 or first > last:
            raise ValueError("Input does not contain a valid JSON object based on '{' and '}'")
        json_str = s[first:last+1]
        return json.loads(json_str)
    except Exception as e:
        print("Parsing JSON failed with error: {}".format(e))
        if default_json:
            return default_json