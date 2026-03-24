import json
import os
import re
import requests
import shlex
import subprocess

from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


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

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, dict] | None = None,
        required: list[str] | None = None,
        *,
        method: str,
    ):
        if parameters is None:
            parameters = ToolParams()
        if required is None:
            required = []
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            required=required,
        )
        old_method = _tool_method_map.get(name, None)
        if old_method and old_method != method:
            raise ValueError(
                f"Redefinition of tool method for {name}: {method} (was {old_method})"
            )
        _tool_method_map[name] = method


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
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GRAY = '\033[90m'
    RED = '\033[31m'
    GREEN = '\033[32m'

    assert(resp.object == 'chat.completion')
    # Assume first choice is correct
    c, *_ = resp.choices
    assert(c.finish_reason is not None)
    m = c.message
    assert(m.role == 'assistant')

    think = m.reasoning_content
    if 0 and len(think) > 120:
        think = think[:80] + '[...]' + think[-30:]

    msg = m.content

    print(f"{GRAY}{m.role}: {c.finish_reason}{RESET}")
    print(f"{GRAY}{think}{RESET}")
    print(msg)


class Agent:
    def __init__(
        self,
        client: Client,
        model: str | None = None,
        model_identity="You are Ostracod, a helpful assistant.",
        system_message: str | None = None,
        tools: list[ToolFunc] | None = None,
        working_dir: Path | str | None = None,
        bin_dir: Path | str | None = None,
    ):
        if system_message is None:
            system_message = model_identity

        available_models = client.models()
        if not model:
            model = list(available_models)[0]

        self.client = client
        self.model_identity = model_identity
        self.system_message = system_message
        self.model_data = available_models[model]
        self.tools = tools or []
        self.working_dir = Path(working_dir or "./workingdir").resolve(True)
        self.bin_dir = Path(bin_dir or "./bin").resolve(True)

        if not (self.working_dir.exists() and self.working_dir.is_dir()):
            raise AgentError(f"Bad working directory given: {self.working_dir}")

        if not (self.bin_dir.exists() and self.bin_dir.is_dir()):
            raise AgentError(f"Bad bin directory given: {self.bin_dir}")

    @property
    def model_name(self):
        return self.model_data.id

    @property
    def model_ctx(self):
        return self.model_data.meta.get("n_ctx_train", 0)

    def basic_completion(self, user_content: str):
        data = CompletionRequest(
            model=self.model_name,
            messages=[
                MsgItem(role="system", content=self.system_message),
                MsgItem(role="user", content=user_content),
            ],
        )
        resp = self.client.single_completion(data)
        return resp.choices[0].message.content

    # TODO: break this up into methods that
    # - Generate msg_item list
    # - Loop on tool calls
    # - Extract final message
    # - Do history...?
    def tool_completion(self, user_content: str, max_rounds=10):
        tool_items = [ToolItem(function=t) for t in self.tools]
        msg_items = [
            MsgItem(role="system", content=self.system_message),
            MsgItem(role="user", content=user_content),
        ]
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
            msg_items.append(resp.message)
            # print(msg_items[-1])
            # print(resp)
            if resp.finish_reason == "tool_calls":
                if max_rounds <= 0:
                    raise AgentError("Too many tool calls")
                for t in resp.message.tool_calls:
                    # TODO handle stuff
                    kwargs = t.function.args_as_kwargs()
                    f = getattr(self, _tool_method_map[t.function.name])
                    r = json.dumps(f(**kwargs))
                    msg_items.append(MsgItem(role="tool", tool_call_id=t.id, content=r))
                    print(msg_items[-1])
                max_rounds -= 1
            elif resp.finish_reason == "stop":
                break
        return resp.message.content

    @property
    def tool_env(self):
        env = {
            "PATH": str(self.bin_dir),
            "GIT_AUTHOR_NAME": "Ostracod",
            "GIT_AUTHOR_EMAIL": "",
        }
        allowed = {"LANG", "LOCPATH", "NLSPATH"}
        for k, v in os.environ.items():
            if k in allowed or k[:3] == "LC_":
                env[k] = v
        return env

    def shell_hack(self, argv):
        s = shlex.shlex(argv, posix=True, punctuation_chars=True)
        s.whitespace_split = True
        arg_list = list(s)

        # NB: we mostly rely on restricted PATH and shell for checks, but also want to have our own checks
        def restricted(reason: str):
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"sh: {reason}: restricted",
            }

        for a in arg_list:
            m = re.match(r"^(>+|PATH=|\.\.|[&])", a)
            if m:
                return restricted(m.group(1))

        if re.match(r"git.config\b.*--global", argv):
            return restricted("git config --global")

        m = re.match(r"\bfind\b.*\b-(exec|execdir|ok|okdir)\b", argv)
        if m:
            return restricted(f"find -{m.group(1)}")

        r = subprocess.run(
            ["/bin/sh", "-r", "-c", argv],
            capture_output=True,
            cwd=self.working_dir,
            env=self.tool_env,
        )
        return {
            "returncode": r.returncode,
            "stdout": r.stdout.decode(),
            "stderr": r.stderr.decode(),
        }

    def write_hack(self, path, content):
        path = (self.working_dir / path).resolve()
        try:
            path.relative_to(self.working_dir)
        except ValueError:
            return {
                "error": "Path not in working directory",
                "bytes_written": 0,
            }

        if "/.git/" in path:
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

    # NB: reads whole file into memory, then writes it out in-place without
    # trying to normalize line endings or whatever
    def update_hack(self, path, content, line_start=-1, replace=False):
        path = (self.working_dir / path).resolve()
        try:
            path.relative_to(self.working_dir)
        except ValueError:
            return {
                "error": "Path not in working directory",
                "lines_written": 0,
                "lines_total": 0,
            }

        if "/.git/" in path:
            return {
                "error": "Can't update files in git repository directly",
                "bytes_written": 0,
            }

        try:
            with path.open("r") as fh:
                lines = fh.readlines()
        except FileNotFoundError:
            lines = []

        if line_start < 0:
            line_start += len(lines) + 1

        if replace and line_start <= 0:
            return {
                "error": "Can't replace lines before start of file",
                "lines_written": 0,
                "lines_total": len(lines),
            }

        content_lines = content.splitlines(True)

        error = None
        lines_written = 0
        lines_total = 0
        try:
            with path.open("w") as fh:
                fh.writelines(lines[:line_start])
                lines_total += line_start
                fh.writelines(content_lines)
                lines_written = len(content_lines)
                lines_total += lines_written
                if replace:
                    line_start += len(content_lines)
                fh.writelines(lines[line_start:])
                lines_written += len(lines[line_start:])
        except IOError as e:
            error = f"Couldn't write lines: {e}"

        return {
            "error": error,
            "lines_written": lines_written,
            "lines_total": lines_total,
        }


if __name__ == "__main__":
    a = Agent(
        Client("http://192.168.19.99:8001/v1"),
        system_message="""You are an interactive agent in a workspace. The workspace might contain git repositories, but it is not itself a git repository.

You have access to some basic tools.

You need to verify the state of the workspace before making changes to it.

You should ask for clarification and guidance if necessary.
""",
        tools=[
            ToolFunc(
                name="shell",
                method="shell_hack",
                description='Execute commands in a restricted POSIX shell. Note that many "dangerous" things (like running a new shell or changing directories) are not allowed.',
                parameters=ToolParams(
                    properties={"argv": PropT("string")},
                    required=["argv"],
                ),
            ),
            ToolFunc(
                name="write_text",
                method="write_hack",
                description="Write `content` to the file at `path`, replacing file contents.",
                parameters=ToolParams(
                    properties={"path": PropT("string"), "content": PropT("string")},
                    required=["path", "content"],
                ),
            ),
            ToolFunc(
                name="update_text",
                method="update_hack",
                description="Write `content` to the file at `path`, optionally specifying a starting line and whether to add or replace lines. Lines are not replaced by default (`replace` is false). Zero offset inserts before first line, negative offsets count backwards from the end of the file. `line_start` defaults to -1, appending to the end of the file.",
                parameters=ToolParams(
                    properties={
                        "path": PropT("string"),
                        "content": PropT("string"),
                        "line_start": PropT("integer"),
                        "replace": PropT("boolean"),
                    },
                    required=["path", "content"],
                ),
            ),
        ],
    )
    # for m in a.client.models():
    #     print(m)
    # r = a.basic_completion("Hello!")
    # r = a.tool_completion("Who are you?")
    # r = a.tool_completion("Create a Python 'Hello, world!' script in the 'hello' repository if one doesn't exist already. Ensure the result is committed to the master branch.")
    # r = a.tool_completion("What would it take to add tests to the 'hello' repo?")
    r = a.tool_completion("Summarize the state of the 'hello' repository.")
    # breakpoint()
    # print(r)
