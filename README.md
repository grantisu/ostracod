# Ostracod

Ostracod is a small AI agent I wrote in Python to better understand how to use and interact with LLMs.

Hopefully it's simple enough to understand most of what's going on,
while still being messy enough to not discourage hacking on the code.

### Usage

The default config assumes that a chat completion server is running locally on port 8001,
since that's what I used for all of my testing:

```
llama-server --host 0.0.0.0 --port 8001 -m something-with-tool-calling-support.gguf [...]
```

Everything else is meant to run inside a container, to limit how much damage can be done.
`Dockerfile` sets up a basic environment and `docker-compose.yaml` sets up a shared workspace and command history, so a single command should set everything up:

```
docker-compose run agent
```

Anything typed at the `>` prompt will get sent to the agent,
unless it starts with a `%` or `/`:
anything after `%` will be sent to a shell,
and `/` commands are handled by the run loop:

```
> Hello!

Hello! I'm Ostracod, ready to help you with your workspace. How can I assist you today?

> %echo "This is a pointless text file." > file.txt

> What's in the workspace?

[thinking, tool calls, etc.]

The workspace contains a single file:

**file.txt**
- Contains: "This is a pointless text file."

That's the entire workspace - just one simple text file.

> /h
Available commands:

/help: show this message.
[...]
/loop PROMPT: send PROMPT in a loop until the model thinks it's done.
```

The `/loop` command can be used for basic vibe coding, e.g.
```
$ docker-compose run -p 8080:8080 agent
[...]
> /loop Build a Python todo web app that stores tasks in sqlite. It should be running in the background, listening on port 8080 for requests. Install dependencies as necessary
Entering loop

[thinking, tool calls, etc.]

Loop completed.
> %python -c "import requests; print(requests.get('http://localhost:8080/'))"
<Response [200]>
```

