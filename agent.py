import json
import sys
from urllib.request import Request, urlopen

BASE_URL='http://192.168.19.99:8001/v1'
BASE_SYS='You are a helpful assistant.'
TOOL_SYS='''You have access to the following tools:

- **shell**: Execute shell commands.
- **calc**: Evaluate basic arithmetic expressions.
- **open**: Opens a gate.
'''

TOOL_COMPL = [{
    'type': 'function',
    'function': {
        'name': 'shell',
        'description': 'Execute shell commands.',
        'parameters': {
            'type': 'object',
            'properties': {'argv': {'type': 'string'}},
            'required': ['argv'],
        }
    }
}, {
    'type': 'function',
    'function': {
        'name': 'open',
        'description': 'Opens a gate.',
        'parameters': {
            'type': 'object',
            'properties': {'argv': {'type': 'string'}},
            'required': ['argv'],
        }
    }
}]

TOOL_RESULTS = {
    'shell': {
        '{"argv":"ls -R"}': '''derp.py
''',
        '{"argv":"ls -l"}': '''total 16
-rw-r--r--   1 gmathews  staff   3842 Feb 23 21:15 derp.py
''',
        '{"argv":"ls -la"}': '''total 40
drwxr-xr-x   4 gmathews  staff    128 Feb 23 21:15 .
drwx------+ 88 gmathews  staff   2816 Feb 23 16:27 ..
-rw-------   1 gmathews  staff  16384 Feb 23 21:27 .derp.py.swp
-rw-r--r--   1 gmathews  staff   3842 Feb 23 21:15 derp.py
''',
        '{"argv":"ls -al"}': '''total 40
drwxr-xr-x   4 gmathews  staff    128 Feb 23 21:15 .
drwx------+ 88 gmathews  staff   2816 Feb 23 16:27 ..
-rw-------   1 gmathews  staff  16384 Feb 23 21:27 .derp.py.swp
-rw-r--r--   1 gmathews  staff   3842 Feb 23 21:15 derp.py
''',
    },
}

def get_basic_completion(user_content: str):
    r = urlopen(BASE_URL + '/chat/completions',
        data=json.dumps({
        "messages": [{
            "role": "system",
            "content": TOOL_SYS,
        }, {
            "role": "user",
            "content": user_content,
        }],
        }).encode())
    return r

def get_tool_completion(user_content: str):
    inp = {
        "chat_template_kwargs": {
            "reasoning_effort": "high",
            "model_identity": "You are Ostracod, a helpful assistant.",
        },
        "messages": [{
            "role": "system",
            "content": TOOL_SYS,
        }, {
            "role": "user",
            "content": user_content,
        }],
        "tools": TOOL_COMPL,
    }
    r = urlopen(BASE_URL + '/chat/completions',
        data=json.dumps(inp).encode())
    d = json.load(r)
    assert(d['object'] == 'chat.completion')
    # Assume first choice is correct
    c, *_ = d['choices']
    #return d
    if c['finish_reason'] == 'tool_calls':
        tc = c['message']['tool_calls'][0]
        #print("Gotta call a tool for:", tc)
        print_basic_completion(d)
        f = tc["function"]
        # ...and put it back in the message list?
        # https://developers.openai.com/api/docs/guides/function-calling
        inp["messages"] += [c["message"], {
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": TOOL_RESULTS[f["name"]][f["arguments"]],
        }]
        r = urlopen(BASE_URL + '/chat/completions', data=json.dumps(inp).encode())
        d = json.load(r)
    return d

def print_basic_completion(d):
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GRAY = '\033[90m'
    RED = '\033[31m'
    GREEN = '\033[32m'

    assert(d['object'] == 'chat.completion')
    # Assume first choice is correct
    c, *_ = d['choices']
    assert(c['finish_reason'] is not None)
    m = c["message"]
    assert(m["role"] == 'assistant')

    think = m["reasoning_content"]
    if 0 and len(think) > 120:
        think = think[:80] + '[...]' + think[-30:]

    msg = m["content"]

    print(f"{GRAY}{think}{RESET}")
    print(msg)
    print(f"{GRAY}{c}{RESET}")

def get_streaming_completion(user_content: str):
    r = Request(BASE_URL + '/chat/completions',
        headers={
            "Accept": "*/*",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "messages": [{
                "role": "system",
                "content": TOOL_SYS,
            }, {
                "role": "user",
                "content": user_content,
            }],
            "stream": True,
        }).encode())
    print(r.headers)
    return urlopen(r)

def print_streaming_completion(r):
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GRAY = '\033[90m'
    RED = '\033[31m'
    GREEN = '\033[32m'

    if r.headers["Content-Type"] != "text/event-stream":
        # Fall back to basic?
        return print_basic_completion(r)

    def data_chunker():
        curline = r.read1()
        while curline:
            while curline[-2:] != b'\n\n':
                frag = r.read1()
                curline += frag
            for chunk in curline.strip(b'\n').split(b'\n\n'):
                if chunk == b'data: [DONE]':
                    return
                yield json.loads(chunk[len('data: '):])["choices"][0]
            curline = r.read1()

    def w(s):
        sys.stdout.write(s)
        sys.stdout.flush()

    reasoning = True
    rlen = 0
    try:
        w(GRAY)
        for co in data_chunker():
            d = co['delta']
            rcontent = d.get('reasoning_content')
            content = d.get('content')
            if rcontent:
                w(rcontent.replace('\n', ' '))
                rlen += len(rcontent)
                if rlen > 120:
                    rlen = 0
                    w('\r')
            elif content:
                if reasoning:
                    reasoning=False
                    print(RESET)
                w(content)
            else:
                print(f'\n\n{GRAY}{co["finish_reason"]}')
    finally:
        print(RESET)


q = "Who are you?"
#q = "Write a Python program to count phonemes in a text file."
#q = "What weighs more: a house cat or a small dog?"
#q = "What's in the current directory?"

if 1:
    r = get_tool_completion(q)
    print_basic_completion(r)
else:
    r = get_streaming_completion(q)
    print(r.headers)
    print_streaming_completion(r)
