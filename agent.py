import json
import os
import re
import requests
import shlex
import subprocess

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from string import Template

from pydantic import BaseModel

try:
    import readline
except ImportError:
    pass


class AgentError(Exception):
    """Something went wrong with running the agent"""


class ModelT(BaseModel):
    """Ensure type member is always set"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_fields_set.add("type")


class ModelData(BaseModel):
    id: str
    meta: dict[str, int]


class PropType(str, Enum):
    string = "string"
    number = "number"
    integer = "integer"
    boolean = "boolean"
    object = "object"
    array = "array"
    null = "null"


class PropT(ModelT):
    type: PropType

    def __init__(self, t):
        super().__init__(type=t)


class FinishReason(str, Enum):
    stop = "stop"
    tool_calls = "tool_calls"


class ToolChoice(str, Enum):
    none = "none"
    auto = "auto"
    required = "required"


class ToolParams(ModelT):
    type: str = "object"
    properties: dict[str, PropT] | None = None
    required: list[str] | None = None


_tool_method_map: dict[str, str] = {}


class ToolFunc(BaseModel):
    name: str
    description: str
    parameters: ToolParams


class ToolDef(BaseModel):
    func: Callable
    meta: ToolFunc

    model_config = {
        "arbitrary_types_allowed": True,
    }


class ToolItem(ModelT):
    type: str = "function"
    function: ToolFunc


class ToolCallFunc(BaseModel):
    name: str
    arguments: str

    def args_as_kwargs(self):
        return json.loads(self.arguments)


class ToolCallItem(ModelT):
    type: str = "function"
    id: str
    function: ToolCallFunc


class MsgRole(str, Enum):
    assistant = "assistant"
    system = "system"
    tool = "tool"
    user = "user"


class MsgItem(BaseModel):
    role: MsgRole
    content: str | None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallItem] | None = None
    tool_call_id: str | None = None


class ChatTemplateKwargs(BaseModel):
    reasoning_effort: str
    model_identity: str


class CompletionRequest(BaseModel):
    model: str | None = None
    messages: list[MsgItem]
    tools: list[ToolItem] | None = None
    tool_choice: ToolChoice | None = None
    chat_template_kwargs: ChatTemplateKwargs | None = None


class ChoiceItem(BaseModel):
    finish_reason: FinishReason | None
    index: int
    message: MsgItem


class UsageItem(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    choices: list[ChoiceItem]
    model: str
    system_fingerprint: str
    object: str
    usage: UsageItem | None = None


class Client:
    """Connect to a completion server, currently just llama.cpp's OpenAI compatibility layer"""

    def __init__(self, base_uri: str):
        self.base_uri = base_uri
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            f"ostracod-agent/0.1 ({self.session.headers['User-Agent']})"
        )

    def models(self):
        return {
            m["id"]: ModelData(**m)
            for m in self.session.get(self.base_uri + "/models").json()["data"]
        }

    def single_completion(self, data: CompletionRequest) -> CompletionResponse:
        resp = self.session.post(
            self.base_uri + "/chat/completions",
            data=data.model_dump_json(exclude_unset=True),
        )
        return CompletionResponse.model_validate_json(resp.content)


def print_basic_completion_response(resp: CompletionResponse):
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GRAY = "\033[90m"
    RED = "\033[31m"
    GREEN = "\033[32m"

    assert resp.object == "chat.completion"
    # Assume first choice is correct
    c, *_ = resp.choices
    assert c.finish_reason is not None
    m = c.message
    assert m.role == "assistant"

    think = m.reasoning_content
    if 0 and len(think) > 120:
        think = think[:80] + "[...]" + think[-30:]

    msg = m.content

    print(f"{GRAY}{m.role}: {c.finish_reason}{RESET}")
    print(f"{GRAY}{think}{RESET}")
    print(msg)


class Agent:
    def __init__(
        self,
        client: Client | str,
        model: str | None = None,
        model_identity="You are Ostracod, a helpful assistant.",
        system_message: str | None = None,
        tools: Iterable[ToolDef] | None = None,
        working_dir: Path | str = "./workingdir",
        bin_dir: Path | str = "./bin",
        config_dir: Path | str = "./configdir",
    ):
        self.config_dir = Path(config_dir).resolve(True)
        self.working_dir = Path(working_dir).resolve(True)
        self.bin_dir = Path(bin_dir).resolve(True)

        self.safe_shell = True
        if self.bin_dir in [Path('/bin'), Path('/usr/bin')]:
            print("DISABLING SHELL SAFETY")
            self.safe_shell = False

        if isinstance(client, str):
            client = Client(client)

        if system_message is None:
            system_message = model_identity

        available_models = client.models()
        if not model:
            model = list(available_models)[0]

        if tools is None:
            tools = self.default_tools

        self.client = client
        self.model_identity = model_identity
        # TODO: t-string templates?
        self.system_template = Template(system_message)
        self.model_data = available_models[model]
        self.tools = {t.meta.name: t for t in tools}
        self.message_history: list[MsgItem] = []

        for prefix in ("bin", "working", "config"):
            path = getattr(self, f"{prefix}_dir")
            if not (path.exists() and path.is_dir()):
                raise AgentError(f"Bad {prefix} directory given: {path}")

    @property
    def model_name(self):
        return self.model_data.id

    @property
    def model_ctx(self):
        return self.model_data.meta.get("n_ctx_train", 0)

    @property
    def system_message(self):
        return self.system_template.substitute(
            now=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        )

    @property
    def default_tools(self):
        return [self.shell_tool, self.write_tool]

    def call_tool(self, t: ToolCallFunc) -> dict:
        kwargs = t.function.args_as_kwargs()
        return self.tools[t.function.name].func(self, **kwargs)

    def tool_completion(self, user_content: str | None, max_rounds=50):
        tool_items = [ToolItem(function=t.meta) for t in self.tools.values()]
        if user_content is not None:
            self.message_history += [MsgItem(role="user", content=user_content)]
        msg_items = [MsgItem(role="system", content=self.system_message)] + self.message_history
        for m in msg_items:
            print(m)
        while True:
            print(f"Start loop: {len(msg_items)} msg items")
            data = CompletionRequest(
                model=self.model_name,
                messages=msg_items,
                tools=tool_items,
                tool_choice=(
                    ToolChoice("auto") if max_rounds > 0 else ToolChoice("none")
                ),
                chat_template_kwargs=ChatTemplateKwargs(
                    reasoning_effort="high", model_identity=self.model_identity
                ),
            )
            completion = self.client.single_completion(data)
            print_basic_completion_response(completion)
            resp = completion.choices[0]
            msg_items += [resp.message]
            # print(msg_items[-1])
            # print(resp)
            if resp.finish_reason == "tool_calls":
                if max_rounds <= 0:
                    raise AgentError("Too many tool calls")
                for t in resp.message.tool_calls:
                    r = json.dumps(self.call_tool(t))
                    msg_items += [MsgItem(role="tool", tool_call_id=t.id, content=r)]
                    print("TOOL", t.function.name, t.function.arguments, msg_items[-1].content)
                max_rounds -= 1
            elif resp.finish_reason == "stop":
                break
        self.message_history += [MsgItem(role="assistant", content=resp.message.content)]
        return resp.message.content or "{ERR: no agent output}"

    @property
    def tool_env(self):
        env = {
            "PATH": str(self.bin_dir),
            "GIT_CONFIG_GLOBAL": self.config_dir / "gitconfig",
        }
        allowed = {"LANG", "LOCPATH", "NLSPATH"}
        for k, v in os.environ.items():
            if k in allowed or k[:3] == "LC_":
                env[k] = v
        return env

    def run_shell_tool(self, argv: str):
        s = shlex.shlex(argv, posix=True, punctuation_chars=True)
        s.whitespace_split = True
        arg_list = list(s)

        if not self.safe_shell:
            r = subprocess.run(
                ["/bin/sh", "-c", argv],
                capture_output=True,
                cwd=self.working_dir,
                timeout=300,
            )
            return {
                "returncode": r.returncode,
                "stdout": r.stdout.decode(),
                "stderr": r.stderr.decode(),
            }

        # NB: we rely on restricted PATH and shell for checks, but also
        # want to have our own checks
        def restricted(reason: str):
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"sh: {reason}: restricted",
            }

        for a in arg_list:
            m = re.match(r"^([<>]+|PATH=|\.\.|(?<!&)&(?!&))", a)
            if m:
                return restricted(m.group(1))

        if re.match(r"git.config\b.*--global", argv):
            return restricted("git config --global")

        m = re.match(r"\bfind\b.*\b-(exec|execdir|ok|okdir)\b", argv)
        if m:
            return restricted(f"find -{m.group(1)}")

        if re.match(r"\bdate\b[^|;&]*\b(-s|--set|(?<=[ '\"])[0-9]{4,8})\b", argv):
            return restricted(f"date --set")

        # TODO: re-allow redirects?
        r = subprocess.run(
            ["/bin/sh", "-r", "-c", argv],
            capture_output=True,
            cwd=self.working_dir,
            env=self.tool_env,
            timeout=30,
        )
        return {
            "returncode": r.returncode,
            "stdout": r.stdout.decode(),
            "stderr": r.stderr.decode(),
        }

    @property
    def shell_tool(self):
        if not self.safe_shell:
            desc = f"Execute commands in a POSIX shell "
        else:
            desc = f"Execute commands in a restricted POSIX shell. "
            'Note that many "dangerous" things (like running a new shell '
            "or changing directories) are not allowed.\n"
            f"PATH contains: {', '.join(f.name for f in self.bin_dir.iterdir())}",

        return ToolDef(
            func=self.__class__.run_shell_tool,
            meta=ToolFunc(
                name="shell",
                description=desc,
                parameters=ToolParams(
                    properties={"argv": PropT("string")},
                    required=["argv"],
                ),
            ),
        )


    def run_write_tool(self, path: str, content: str):
        path = (self.working_dir / path).resolve()
        try:
            path.relative_to(self.working_dir)
        except ValueError:
            return {
                "error": "Path not in working directory",
                "bytes_written": 0,
            }

        if "/.git/" in str(path):
            return {
                "error": "Can't write to files in git repository directly",
                "bytes_written": 0,
            }

        error = None
        bytes_written = 0
        try:
            with path.open("w") as fh:
                bytes_written = fh.write(content)
        except IOError as e:
            error = f"Couldn't write: {e}"

        return {
            "error": error,
            "bytes_written": bytes_written,
        }

    @property
    def write_tool(self):
        return ToolDef(
            func=self.__class__.run_write_tool,
            meta=ToolFunc(
                name="write",
                description="Write `content` to the file at `path`, replacing "
                "any existing file contents.",
                parameters=ToolParams(
                    properties={
                        "path": PropT("string"),
                        "content": PropT("string"),
                    },
                    required=["path", "content"],
                ),
            ),
        )


# TODO: normalize I/O
if __name__ == "__main__":
    import tomllib
    with Path('./configdir/agent.toml').open('rb') as fh:
        agent_config = tomllib.load(fh)["Agent"]
    a = Agent(**agent_config)
    try:
        while True:
            inp = input("> ")
            if inp[:1] == "/":
                # Driver commands
                if inp == '/empty':
                    a.tool_completion(None)
                elif inp == '/history'[:len(inp)]:
                    for msg in a.message_history:
                        print(msg)
                else:
                    print(f"Unknown command: {inp}")
            elif inp[:1] == "%":
                # Shell commands
                r = a.run_shell_tool(inp[1:])
                print(r["stderr"])
                print(r["stdout"])
            else:
                a.tool_completion(inp)
    except EOFError:
        pass
