import io
import json
import os
import re
import requests
import shlex
import subprocess
import sys

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, IO

from pydantic import BaseModel, JsonValue

RESET = "\033[0m"
BOLD = "\033[1m"
GRAY = "\033[90m"
RED = "\033[31m"
GREEN = "\033[32m"


class AgentError(Exception):
    """Something went wrong with running the agent"""


class ModelT(BaseModel):
    """Ensure type member is always set"""

    def __init__(self, *args: Any, **kwargs: Any):
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

    def __init__(self, t: str):
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


class ToolFunc(BaseModel):
    name: str
    description: str
    parameters: ToolParams


class ToolDef(BaseModel):
    func: Callable[..., dict[str, str | int | None]]
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

    def args_as_kwargs(self) -> Any:
        return json.loads(self.arguments)


class ToolCallItem(ModelT):
    type: str = "function"
    id: str
    function: ToolCallFunc


class Url(BaseModel):
    url: str


class TypedContent(BaseModel):
    type: str
    text: str | None = None
    image_url: Url | None = None


class MsgRole(str, Enum):
    assistant = "assistant"
    system = "system"
    tool = "tool"
    user = "user"


class MsgItem(BaseModel):
    role: MsgRole
    content: str | list[TypedContent] | None
    reasoning_content: str | None = None
    tool_calls: list[ToolCallItem] | None = None
    tool_call_id: str | None = None

    def __init__(self, role: str | MsgRole, **kwargs: Any):
        if isinstance(role, str):
            role = MsgRole(role)
        super().__init__(role=role, **kwargs)


class ChatTemplateKwargs(BaseModel):
    reasoning_effort: str
    model_identity: str
    enable_thinking: bool


class CompletionRequest(BaseModel):
    model: str | None = None
    messages: list[MsgItem]
    tools: list[ToolItem] | None = None
    tool_choice: ToolChoice | None = None
    chat_template_kwargs: ChatTemplateKwargs | None = None
    stream: bool = False
    response_format: JsonValue = None
    n: int = 1


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

    def models(self) -> dict[str, ModelData]:
        return {
            m["id"]: ModelData(**m)
            for m in self.session.get(self.base_uri + "/models").json()["data"]
        }

    def single_completion(
        self, data: CompletionRequest, stdout: IO[str] = sys.stdout
    ) -> CompletionResponse:
        resp = self.session.post(
            self.base_uri + "/chat/completions",
            data=data.model_dump_json(exclude_unset=True),
            stream=data.stream,
        )
        if resp.headers["Content-Type"] == "text/event-stream":
            return self.assemble_from_stream(resp, stdout=stdout)
        else:
            resp_data = CompletionResponse.model_validate_json(resp.content)
            fin = resp_data.choices[0].finish_reason
            msg = resp_data.choices[0].message
            if msg.reasoning_content:
                stdout.write(f"{GRAY}{msg.reasoning_content}{RESET}\n")
            if msg.content:
                stdout.write(f"{msg.content}\n")
            if fin:
                stdout.write(f"{GRAY}{fin}{RESET}")
            return resp_data

    # TODO: cleanup
    def assemble_from_stream(
        self,
        resp: requests.models.Response,
        stdout: IO[str] = sys.stdout,
    ) -> CompletionResponse:
        def data_chunker() -> Iterator[dict[str, Any]]:
            for line in resp.iter_lines(chunk_size=None, delimiter=b"\n\n"):
                if line == b"data: [DONE]":
                    return
                if line:
                    yield json.loads(line[len("data: ") :])
            raise ValueError("Missing DONE line")

        def w(s: str) -> None:
            stdout.write(s)
            stdout.flush()

        reasoning = True
        tooling = False
        rfrag = []
        cfrag = []
        role = None
        finish_reason = None
        index = 0  # XXX Assume first response is what we want
        tfrags: dict[int, ToolCallItem] = {}
        try:
            w(GRAY)
            for chunk in data_chunker():
                co = chunk["choices"][0]
                finish_reason = co.get("finish_reason")
                d = co["delta"]
                if "role" in d:
                    role = MsgRole(d["role"])
                rcontent = d.get("reasoning_content")
                content = d.get("content")
                tool_calls = d.get("tool_calls", ())
                if rcontent:
                    w(rcontent)
                    rfrag.append(rcontent)
                if content:
                    if reasoning:
                        reasoning = False
                        w(f"{RESET}\n")
                    w(content)
                    cfrag.append(content)
                if tool_calls and not tooling:
                    tooling = True
                    w(RED)
                if tooling and not tool_calls:
                    tooling = False
                    w(RESET)
                for t_delta in tool_calls:
                    tool = tfrags.get(t_delta["index"])
                    if tool is None:
                        tool = ToolCallItem.model_validate(t_delta)
                        w(f"\n{tool.function.name} {tool.function.arguments}")
                        tfrags[t_delta["index"]] = tool
                    else:
                        arg_delta = t_delta["function"]["arguments"]
                        tool.function.arguments += arg_delta
                        w(arg_delta)
                if co["finish_reason"]:
                    w(f'\n{GRAY}{co["finish_reason"]}\n')
        finally:
            w(f"{RESET}\n")

        assert role is not None

        # Build aggregate response off of final chunk
        m = MsgItem(role=role, content="".join(cfrag))
        if rfrag:
            m.reasoning_content = "".join(rfrag)
        if tfrags:
            m.tool_calls = list(tfrags.values())

        return CompletionResponse.model_validate(
            {
                **chunk,
                "choices": [
                    ChoiceItem(finish_reason=finish_reason, index=index, message=m)
                ],
            }
        )


class Agent:
    def __init__(
        self,
        client: Client | str,
        model: str | None = None,
        model_identity: str = "You are Ostracod, a helpful assistant.",
        reasoning_effort: str = "medium",
        enable_thinking: bool = True,
        system_message: str | None = None,
        tools: Iterable[ToolDef] | None = None,
        working_dir: Path | str = "./workingdir",
        bin_dir: Path | str = "./bin",
        config_dir: Path | str = "./configdir",
        stdout: IO[str] = sys.stdout,
    ):
        self.config_dir = Path(config_dir).resolve(True)
        self.working_dir = Path(working_dir).resolve(True)
        self.bin_dir = Path(bin_dir).resolve(True)

        self.safe_shell = True
        if self.bin_dir in [Path("/bin"), Path("/usr/bin")]:
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
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        # TODO: t-string templates?
        self.system_template = Template(f"{model_identity}\n{system_message}")
        self.model_data = available_models[model]
        self.tools = {t.meta.name: t for t in tools}
        self.message_history: list[MsgItem] = []
        self.stdout = stdout

        for prefix in ("bin", "working", "config"):
            path = getattr(self, f"{prefix}_dir")
            if not (path.exists() and path.is_dir()):
                raise AgentError(f"Bad {prefix} directory given: {path}")

    @property
    def model_name(self) -> str:
        return self.model_data.id

    @property
    def model_ctx(self) -> int:
        return self.model_data.meta.get("n_ctx_train", 0)

    @property
    def system_message(self) -> str:
        return self.system_template.substitute(
            now=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        )

    @property
    def chat_template_kwargs(self) -> ChatTemplateKwargs:
        return ChatTemplateKwargs(
            model_identity=self.model_identity,
            reasoning_effort=self.reasoning_effort,
            enable_thinking=self.enable_thinking,
        )

    @property
    def default_tools(self) -> list[ToolDef]:
        return [self.shell_tool, self.write_tool]

    def call_tool(self, t: ToolCallItem) -> Any:
        kwargs = t.function.args_as_kwargs()
        return self.tools[t.function.name].func(self, **kwargs)

    def basic_streaming_completion(self, user_content: str) -> str:
        data = CompletionRequest(
            model=self.model_name,
            messages=[
                MsgItem(role="system", content=self.system_message),
                MsgItem(role="user", content=user_content),
            ],
            chat_template_kwargs=self.chat_template_kwargs,
            stream=True,
        )
        resp = self.client.single_completion(data, stdout=self.stdout)
        return resp.choices[0].message.content or ""

    def tool_completion(self, user_content: str | None, stdout: IO[str] = sys.stdout, max_rounds: int = 50) -> str:
        tool_items = [ToolItem(function=t.meta) for t in self.tools.values()]
        if user_content is not None:
            self.message_history += [MsgItem(role="user", content=user_content)]
        msg_items = [
            MsgItem(role="system", content=self.system_message)
        ] + self.message_history
        tool_rounds = 0
        while True:
            # print(f"Start loop: {len(msg_items)} msg items")
            data = CompletionRequest(
                model=self.model_name,
                messages=msg_items,
                tools=tool_items,
                tool_choice=(
                    ToolChoice("auto") if tool_rounds < max_rounds else ToolChoice("none")
                ),
                chat_template_kwargs=self.chat_template_kwargs,
                stream=True,
            )
            completion = self.client.single_completion(data, stdout=self.stdout)
            resp = completion.choices[0]
            msg_items += [resp.message]
            # print(msg_items[-1])
            # print(resp)
            if resp.finish_reason == "tool_calls":
                if max_rounds <= tool_rounds:
                    raise AgentError("Too many tool calls")
                for t in resp.message.tool_calls or []:
                    r = json.dumps(self.call_tool(t))
                    self.stdout.write(f"{GRAY}{r}{RESET}\n")
                    msg_items += [MsgItem(role="tool", tool_call_id=t.id, content=r)]
                tool_rounds += 1
            elif resp.finish_reason == "stop":
                final_result = resp.message.content
                break
        self.message_history += [
            MsgItem(role="assistant", content=final_result)
        ]
        return final_result or "{ERR: no agent output}"

    @property
    def tool_env(self) -> dict[str, str]:
        env = {
            "PATH": str(self.bin_dir),
            "GIT_CONFIG_GLOBAL": str(self.config_dir / "gitconfig"),
        }
        allowed = {"LANG", "LOCPATH", "NLSPATH"}
        for k, v in os.environ.items():
            if k in allowed or k[:3] == "LC_":
                env[k] = v
        return env

    def run_shell_tool(self, argv: str, stdin: str = '') -> dict[str, str | int | None]:
        s = shlex.shlex(argv, posix=True, punctuation_chars=True)
        s.whitespace_split = True
        try:
            arg_list = list(s)
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Unparsable input: {e}",
            }

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
        def restricted(reason: str) -> dict[str, str | int | None]:
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
            input=stdin,
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
    def shell_tool(self) -> ToolDef:
        if not self.safe_shell:
            desc = f"Execute commands in a POSIX shell."
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
                    properties={"argv": PropT("string"), "stdin": PropT("string")},
                    required=["argv"],
                ),
            ),
        )

    def run_write_tool(self, path: str, content: str) -> dict[str, str | int | None]:
        rpath = (self.working_dir / path).resolve()
        try:
            rpath.relative_to(self.working_dir)
        except ValueError:
            return {
                "error": "Path not in working directory",
                "bytes_written": 0,
            }

        if "/.git/" in str(rpath):
            return {
                "error": "Can't write to files in git repository directly",
                "bytes_written": 0,
            }

        error = None
        bytes_written = 0
        try:
            with rpath.open("w") as fh:
                bytes_written = fh.write(content)
        except IOError as e:
            error = f"Couldn't write: {e}"

        return {
            "error": error,
            "bytes_written": bytes_written,
        }

    @property
    def write_tool(self) -> ToolDef:
        return ToolDef(
            func=self.__class__.run_write_tool,
            meta=ToolFunc(
                name="write",
                description="Write `content` to the file at `path`, "
                "replacing any existing file contents. Be careful to "
                "not double-escape characters.",
                parameters=ToolParams(
                    properties={
                        "path": PropT("string"),
                        "content": PropT("string"),
                    },
                    required=["path", "content"],
                ),
            ),
        )


# TODO: normalize I/O?
if __name__ == "__main__":
    import atexit
    import readline
    import tomllib

    readline.parse_and_bind("tab: complete")
    rl_history = Path(".derp_history").resolve()
    try:
        readline.read_history_file(rl_history)
        rlh_len = readline.get_current_history_length()
    except FileNotFoundError:
        rl_history.open("wb").close()
        rlh_len = 0

    def rlh_save(prev_rlh_len: int) -> None:
        new_rlh_len = readline.get_current_history_length()
        readline.set_history_length(1000)
        readline.append_history_file(new_rlh_len - prev_rlh_len, rl_history)

    atexit.register(rlh_save, rlh_len)

    with Path("./configdir/agent.toml").open("rb") as fh:
        agent_config = tomllib.load(fh)["Agent"]

    a = Agent(**agent_config)
    os.chdir(a.working_dir)
    try:
        while True:
            inp = input("> ")
            if inp[:1] == "/":
                # Driver commands
                if inp == "/empty":
                    a.tool_completion(None)
                elif inp == "/history"[: len(inp)]:
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
