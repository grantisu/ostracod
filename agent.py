import atexit
import io
import json
import mimetypes
import os
import re
import readline
import requests
import subprocess
import sys

from base64 import b64encode
from collections.abc import Callable, Generator, Iterable, Iterator
from datetime import datetime
from enum import Enum
from functools import cached_property
from pathlib import Path
from string import Template
from typing import Any, IO

from PIL import Image, ImageOps
from pydantic import BaseModel, JsonValue


class AgentError(Exception):
    """Something went wrong with running the agent"""


class ModelT(BaseModel):
    """Ensure type member is always set"""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model_fields_set.add("type")


class ModelData(BaseModel):
    id: str
    meta: dict[str, int | str]


class PropType(str, Enum):
    string = "string"
    number = "number"
    integer = "integer"
    boolean = "boolean"
    object = "object"
    array = "array"
    null = "null"


class ToolProp(ModelT):
    type: PropType
    description: str | None = None

    def __init__(self, t: str, description: str = ""):
        kwargs = {"type": t}
        if description:
            kwargs["description"] = description
        super().__init__(**kwargs)


class FinishReason(str, Enum):
    stop = "stop"
    tool_calls = "tool_calls"


class ToolChoice(str, Enum):
    none = "none"
    auto = "auto"
    required = "required"


class ToolParams(ModelT):
    type: str = "object"
    properties: dict[str, ToolProp] | None = None
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


class MsgRole(str, Enum):
    assistant = "assistant"
    system = "system"
    tool = "tool"
    user = "user"


class Url(BaseModel):
    url: str


class MsgContent(BaseModel):
    type: str
    text: str | None = None
    image_url: Url | None = None


class MsgItem(BaseModel):
    role: MsgRole
    content: str | list[MsgContent] | None
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
    temperature: float | None = None
    n: int = 1


class ChoiceItem(BaseModel):
    finish_reason: FinishReason | None
    index: int
    message: MsgItem


class CompletionResponse(BaseModel):
    choices: list[ChoiceItem]
    model: str
    system_fingerprint: str
    object: str


class StreamingResponse:
    def __init__(self, stream: Generator[tuple[Any, Any, Any, Any], None, CompletionResponse]):
        self._stream = stream
        self.response: CompletionResponse | None = None

    def __iter__(self) -> Iterator[tuple[Any, Any, Any, Any]]:
        try:
            self.response = yield from self._stream
        except ValueError as e:
            raise AgentError(e)


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

    def basic_completion(self, data: CompletionRequest) -> CompletionResponse:
        if data.stream:
            raise ValueError("Streaming request")
        resp = self.session.post(
            self.base_uri + "/chat/completions",
            data=data.model_dump_json(exclude_unset=True),
            stream=False,
        )
        if resp.status_code != 200:
            raise AgentError(resp.json())
        return CompletionResponse.model_validate_json(resp.content)

    def _streaming_completion(
        self, data: CompletionRequest
    ) -> Generator[tuple[Any, Any, str | None, Any], None, CompletionResponse]:
        resp = self.session.post(
            self.base_uri + "/chat/completions",
            data=data.model_dump_json(exclude_unset=True),
            stream=True,
        )
        if resp.headers.get("Content-Type", "") != "text/event-stream":
            resp_msg = resp.json().get("error", {}).get("message", "")
            if resp.status_code in (400, 403, 404, 410):
                err_msg = "Bad Request"
            elif resp.status_code >= 500:
                err_msg = "Server Error"
            else:
                err_msg = "Non-streaming Response"
            raise AgentError(f"{err_msg}: [{resp.status_code}] {resp_msg}")

        def data_chunker() -> Iterator[dict[str, Any]]:
            for line in resp.iter_lines(chunk_size=None, delimiter=b"\n\n"):
                if not line:
                    continue
                elif line == b"data: [DONE]":
                    return
                yield json.loads(line[6:])

            raise AgentError("Missing DONE line")

        reason_frag = []
        content_frag = []
        role = None
        finish_reason = None
        tool_frags: dict[int, ToolCallItem] = {}

        for chunk in data_chunker():
            try:
                co = chunk["choices"][0]
            except KeyError:
                raise AgentError(f"Missing choices in {chunk}")

            finish_reason = co.get("finish_reason")
            d = co["delta"]

            if "role" in d:
                role = MsgRole(d["role"])

            reasoning_content = d.get("reasoning_content")
            content = d.get("content")
            tool_calls = d.get("tool_calls", ())
            tool_delta = None

            if reasoning_content:
                reason_frag.append(reasoning_content)
            if content:
                content_frag.append(content)
            for t_delta in tool_calls:
                tool = tool_frags.get(t_delta["index"])
                if tool is None:
                    tool = ToolCallItem.model_validate(t_delta)
                    tool_frags[t_delta["index"]] = tool
                    tool_delta = f"{tool.function.name} {tool.function.arguments}"
                else:
                    arg_delta = t_delta["function"]["arguments"]
                    tool.function.arguments += arg_delta
                    tool_delta = arg_delta

            yield content, reasoning_content, tool_delta, finish_reason

        assert role is not None

        # Build aggregate response off of final chunk
        m = MsgItem(role=role, content="".join(content_frag))
        if reason_frag:
            m.reasoning_content = "".join(reason_frag)
        if tool_frags:
            m.tool_calls = list(tool_frags.values())

        return CompletionResponse.model_validate(
            {
                **chunk,
                "choices": [ChoiceItem(finish_reason=finish_reason, index=0, message=m)],
            }
        )

    def streaming_completion(self, data: CompletionRequest) -> StreamingResponse:
        if not data.stream:
            raise ValueError("Not a streaming request")
        return StreamingResponse(self._streaming_completion(data))


class Console:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GRAY = "\033[90m"
    RED = "\033[31m"
    GREEN = "\033[32m"

    def __init__(
        self,
        stdin: IO[str] = sys.stdin,
        stdout: IO[str] = sys.stdout,
        color: bool = True,
        history_file: Path | str | None = None,
        prompt: str = "> ",
    ):
        self.stdin = stdin
        self.stdout = stdout
        self.color = color
        self.prompt = prompt
        self.history_file: Path | None = None
        if history_file:
            self.history_file = Path(history_file).resolve()
        self.in_reasoning = False
        self.in_tooling = False
        self._nfrag = 0

        if self.color:
            atexit.register(self.reset)

        if self.history_file is not None:
            if self.history_file.is_dir():
                # TODO: fix this elsewhere
                self.history_file /= "history"
            readline.parse_and_bind("tab: complete")
            try:
                readline.read_history_file(self.history_file)
                rlh_len = readline.get_current_history_length()
            except FileNotFoundError:
                self.history_file.open("wb").close()
                rlh_len = 0

            def rlh_save(prev_rlh_len: int) -> None:
                new_rlh_len = readline.get_current_history_length()
                readline.set_history_length(1000)
                readline.append_history_file(new_rlh_len - prev_rlh_len, self.history_file)

            atexit.register(rlh_save, rlh_len)

    def reset(self, sep: bool = True) -> "Console":
        self.stdout.write(self.RESET)
        self.in_reasoning = False
        self.in_tooling = False
        if sep:
            self.sep()
        return self

    def dim(self, text: str = "") -> "Console":
        self.stdout.write(f"{self.GRAY}{text}")
        return self

    def bright(self, text: str = "") -> "Console":
        self.stdout.write(f"{self.RED}{text}")
        return self

    def sep(self) -> "Console":
        self.stdout.write("\n")
        return self

    def input(self) -> str:
        return input(self.prompt)

    def output(self, s: str) -> "Console":
        self.stdout.write(s)
        return self

    def flush(self) -> "Console":
        self.stdout.flush()
        return self

    # TODO: better convention for multiple channels?
    def emit_fragment(
        self, content: str | None, reasoning: str | None, tooling: Any, stop: Any
    ) -> "Console":
        if reasoning:
            if not self.in_reasoning:
                self.dim().sep()
                self.in_reasoning = True
                self.in_tooling = False
            self.output(reasoning)
        if tooling:
            if not self.in_tooling:
                self.bright().sep()
                self.in_reasoning = False
                self.in_tooling = True
            self.output(str(tooling))
        if content:
            if self.in_reasoning or self.in_tooling:
                self.reset()
            self.output(content)
        self._nfrag += 1
        if (self._nfrag & 3) == 0:
            self.flush()
        return self


class Agent:
    def __init__(
        self,
        client: Client | str,
        console: Console | None = None,
        model: str | None = None,
        model_identity: str = "You are a helpful assistant.",
        reasoning_effort: str = "medium",
        enable_thinking: bool = True,
        system_message: str | None = None,
        temperature: float | None = None,
    ):
        self.console = console or Console()

        if isinstance(client, str):
            client = Client(client)

        if system_message is None:
            system_message = model_identity

        available_models = client.models()
        if not model:
            model = list(available_models)[0]

        self.client = client
        self.model_identity = model_identity
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        # TODO: t-string templates?
        self.system_template = Template(f"{model_identity}\n{system_message}")
        self.model_data = available_models[model]
        self.message_history: list[MsgItem] = []

    @property
    def model_name(self) -> str:
        return self.model_data.id

    # TODO: figure out how to handle limited context size
    @property
    def model_ctx(self) -> int:
        return int(self.model_data.meta.get("n_ctx_train", 0))

    def system_template_args(self) -> dict[str, str]:
        return {
            "now": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        }

    @property
    def system_message(self) -> str:
        return self.system_template.substitute(self.system_template_args())

    @property
    def chat_template_kwargs(self) -> ChatTemplateKwargs:
        return ChatTemplateKwargs(
            model_identity=self.model_identity,
            reasoning_effort=self.reasoning_effort,
            enable_thinking=self.enable_thinking,
        )

    def streaming_completion(self, user_content: str | None, max_rounds: int = 50) -> str:
        if user_content is not None:
            self.message_history += [MsgItem(role="user", content=user_content)]

        msg_items = [MsgItem(role="system", content=self.system_message)] + self.message_history

        data = CompletionRequest(
            model=self.model_name,
            messages=msg_items,
            chat_template_kwargs=self.chat_template_kwargs,
            stream=True,
        )
        if self.temperature is not None:
            data.temperature = self.temperature
        completion = self.client.streaming_completion(data)
        for frag in completion:
            self.console.emit_fragment(*frag)
        self.console.sep()
        assert completion.response is not None
        final_result = completion.response.choices[0].message.content
        self.message_history += [MsgItem(role="assistant", content=final_result)]
        assert isinstance(final_result, str)
        return final_result

    def subshell_helper(
        self,
        argv: list[str],
        timeout: int = 30,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, str | int | None]:
        if env:
            env = {**os.environ.copy(), **env}
        try:
            r = subprocess.run(argv, capture_output=True, timeout=timeout, cwd=cwd, env=env)
        except subprocess.TimeoutExpired as e:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": str(e),
            }

        return {
            "returncode": r.returncode,
            "stdout": r.stdout.decode(errors="replace"),
            "stderr": r.stderr.decode(errors="replace"),
        }

    def summarize_session(self) -> str:
        s_msg = """Summarize the current session so that it can be resumed without disruption later.

Omit the first system message (it will be included verbatim when resumed), but make a concise and accurate summary of all other messages. Each summary must include:

- All topics discussed (major topics should get a sentence or two; minor topics can be summarized as one or two words in a list of topics covered)
- All tasks performed (including which tools were used, but not including full tool output; relevant snippets can be included)
- The _results_ of all tasks performed
- All new information uncovered
- All unresolved questions
- Anything else necessary to seamlessly continue the session (if it's not in the first system message or the summary, then it will be forgotten!)

The summary should be suitable to include as a system message by itself, e.g. by starting with an explanation followed by a list of (condensed) messages:

    This is a continuation of an existing session; resume from where the original left off. Here is a highly abridged record of the session so far:

    User: Hello
    Agent: How can I help you today?
    User: What country has the biggest capital city?
    Agent: [I need to look up the largest capital by population.
      shell {"command":"w3m -dump https://en.wikipedia.org/wiki/List_of_national_capitals_by_population | grep -E \\"Population|City\\" | head -n 20"}]
    Agent: Wikipedia lists Mexico City as the largest by population.
"""

        msg_items = (
            [MsgItem(role="system", content=self.system_message)]
            + self.message_history
            + [MsgItem(role="system", content=s_msg)]
        )

        data = CompletionRequest(
            model=self.model_name,
            messages=msg_items,
            chat_template_kwargs=self.chat_template_kwargs,
            stream=True,
        )
        if self.temperature is not None:
            data.temperature = self.temperature
        completion = self.client.streaming_completion(data)
        for frag in completion:
            self.console.emit_fragment(*frag)
        self.console.sep()
        assert completion.response is not None
        final_result = completion.response.choices[0].message.content
        assert isinstance(final_result, str)
        return final_result

    def run(self) -> None:
        loop_prompt = ""
        loop_until = "done"
        last_output = ""
        while True:
            if loop_prompt:
                inp = loop_prompt
                if last_output.strip().lower() == loop_until.lower():
                    self.console.bright("Loop completed.").reset()
                    loop_until = ""
                    loop_prompt = ""

            if loop_prompt:
                self.console.output(inp).sep()
            else:
                inp = self.console.input()

            if inp[:1] == "/":
                # Driver commands
                cmd, *args = inp.split()
                if cmd == "/help"[: len(cmd)]:
                    self.console.output("""Available commands:

/help: show this message.
/messages: show past messages that are included in completion context.
/clear N: clear last N messages; defaults to removing all messages
/squeeze: replace message history with a summary to free up context
/temperature T: set the temperature to T (should be 0.0 - 2.0, but those limits aren't enforced)
/think O: set thinking on or off
/loop PROMPT: send PROMPT in a loop until the model thinks it's done.
""")
                elif cmd == "/messages"[: len(cmd)]:
                    for msg in self.message_history:
                        self.console.output(str(msg))
                elif cmd == "/clear"[: len(cmd)]:
                    n = orig_len = len(self.message_history)
                    if args and args[0].isdigit():
                        n = int(args[0])
                    self.message_history = self.message_history[:-n]
                    self.console.output(f"Cleared {orig_len - len(self.message_history)} messages")
                elif cmd == "/squeeze"[: len(cmd)]:
                    # Save last interaction to smooth the transition
                    saved_messages = []
                    while self.message_history and self.message_history[-1].role != "user":
                        saved_messages.append(self.message_history.pop())
                    saved_messages.append(self.message_history.pop())
                    saved_messages.reverse()
                    summary = self.summarize_session()
                    self.message_history = [
                        MsgItem(role="system", content=summary)
                    ] + saved_messages
                    self.console.bright("Squeezed session down.").reset()
                elif cmd == "/temperature"[: len(cmd)]:
                    try:
                        self.temperature = float(args[0])
                    except (IndexError, ValueError):
                        self.console.bright(
                            f"Bad temperature args: {args!r}\nCurrent temp is: {self.temperature}"
                        ).reset()
                elif cmd == "/think"[: len(cmd)]:
                    try:
                        v = None
                        if args[0].lower() in "on y yes enable t true 1".split():
                            v = True
                        if args[0].lower() in "off n no disable f false 0".split():
                            v = False
                    except (IndexError, ValueError):
                        pass
                    if v is not None:
                        self.enable_thinking = v
                    else:
                        self.console.bright(
                            f"Bad think args: {args!r}\nCurrent thinking is: {self.enable_thinking}"
                        ).reset()
                elif inp[:6] == "/loop ":
                    self.console.bright("Entering loop").reset()
                    loop_prompt = inp[6:]
                    loop_prompt += '\nIf the task is complete, then say "done" with no other preface or formatting.'
                else:
                    self.console.output(f"Unknown command: {cmd}")
            elif inp[:1] == "%":
                r = self.subshell_helper(["/bin/sh", "-c", inp[1:]])
                self.console.output(str(r["stderr"]))
                self.console.output(str(r["stdout"]))
            else:
                # If we're in a loop, retry things a few times before giving up
                for i in range(3):
                    try:
                        last_output = self.streaming_completion(inp)
                    except KeyboardInterrupt:
                        loop_prompt = ""
                        self.console.bright("Interrupted").reset()
                        break
                    except AgentError as e:
                        if str(e).startswith("Bad Request"):
                            self.console.bright("Removing history up to last user request").reset()
                            while msg := self.message_history.pop():
                                if msg.role == "user":
                                    break
                            break
                        elif not loop_prompt:
                            raise
                    else:
                        break

            self.console.sep()


class ToolAgent(Agent):
    def __init__(
        self,
        client: Client | str,
        console: Console | None = None,
        model: str | None = None,
        model_identity: str = "You are a helpful assistant.",
        reasoning_effort: str = "medium",
        enable_thinking: bool = True,
        system_message: str | None = None,
        temperature: float | None = None,
        tools: Iterable[ToolDef] | None = None,
        working_dir: Path | str = "./workingdir",
        user_agent: str = "ostracod/0.0",
    ):
        self.working_dir = Path(working_dir).resolve(True)
        self.user_agent = user_agent

        super().__init__(
            client=client,
            console=console,
            model=model,
            model_identity=model_identity,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
            system_message=system_message,
            temperature=temperature,
        )

        if tools is None:
            tools = self.default_tools

        self.tools = {t.meta.name: t for t in tools}
        # TODO: figure out better abstraction?
        self.message_queue: list[MsgItem] = []

        if "shell" in self.tools:
            self.console.bright(
                "WARNING: shell runs arbitrary commands; make sure you want this!"
            ).reset()

        if not (self.working_dir.exists() and self.working_dir.is_dir()):
            raise AgentError(f"Bad working directory given: {self.working_dir}")

    def system_template_args(self) -> dict[str, str]:
        targs = super().system_template_args()
        targs["working_dir"] = str(self.working_dir)
        fmt_list = (", " if len(self.tools) != 2 else " ").join(f"`{t}`" for t in self.tools)
        targs["fmt_tool_list"] = re.sub(r" `[^`]+`$", r" and\g<0>", fmt_list)
        return targs

    @cached_property
    def has_mmproj(self) -> bool:
        # TODO: more efficient check?
        try:
            image_url = Url(
                url="data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAAAAAA6mKC9AAAAAXNSR0IB2cksfwAAAARnQU1BAACx"
                "jwv8YQUAAAAgY0hSTQAAeiYAAICEAAD6AAAAgOgAAHUwAADqYAAAOpgAABdwnLpRPAAAAA1JREFU"
                "GNNjYBgFyAAAARAAAeyX43oAAAAASUVORK5CYII="
            )
            self.client.basic_completion(
                CompletionRequest(
                    chat_template_kwargs=ChatTemplateKwargs(
                        model_identity="You are succinct.",
                        enable_thinking=False,
                        reasoning_effort="low",
                    ),
                    messages=[
                        MsgItem(
                            role="user",
                            content=[
                                MsgContent(type="text", text="Describe this image in one word."),
                                MsgContent(
                                    type="image_url",
                                    image_url=image_url,
                                ),
                            ],
                        )
                    ],
                )
            )
        except AgentError as e:
            return False
        return True

    @property
    def default_tools(self) -> list[ToolDef]:
        return [self.read_tool, self.write_tool, self.shell_tool, self.patch_tool, self.web_tool]

    def call_tool(self, t: ToolCallItem) -> Any:
        kwargs = t.function.args_as_kwargs()
        return self.tools[t.function.name].func(self, **kwargs)

    def streaming_completion(self, user_content: str | None, max_rounds: int = 250) -> str:
        if user_content is not None:
            self.message_history += [MsgItem(role="user", content=user_content)]
        msg_items = [MsgItem(role="system", content=self.system_message)] + self.message_history
        tool_items = [ToolItem(function=t.meta) for t in self.tools.values()]
        tool_rounds = 0
        while True:
            data = CompletionRequest(
                model=self.model_name,
                messages=msg_items,
                chat_template_kwargs=self.chat_template_kwargs,
                tool_choice=ToolChoice("none"),
                stream=True,
            )
            if self.temperature is not None:
                data.temperature = self.temperature
            if tool_items and tool_rounds < max_rounds:
                data.tools = tool_items
                data.tool_choice = ToolChoice("auto")

            last_error = None
            for retry in range(3):
                if retry > 0:
                    self.console.bright("RETRYING").reset()
                try:
                    completion = self.client.streaming_completion(data)
                    for frag in completion:
                        self.console.emit_fragment(*frag)
                except AgentError as e:
                    self.console.bright(f"FAILED COMPLETION:\n{e}").reset()
                    last_error = e
                    if str(e).startswith("Bad Request"):
                        break
                else:
                    last_error = None
                    break
            self.console.sep()

            if last_error is not None:
                raise last_error

            assert completion.response is not None
            resp = completion.response.choices[0]
            msg_items += [resp.message]
            self.message_history += msg_items[-1:]
            if resp.finish_reason == "tool_calls":
                if max_rounds <= tool_rounds:
                    raise AgentError("Too many tool calls")
                for t in resp.message.tool_calls or []:
                    tool_result = self.call_tool(t)
                    r: str | list[MsgContent]
                    if isinstance(tool_result, MsgContent):
                        r = [tool_result]
                    else:
                        r = json.dumps(self.call_tool(t))
                    self.console.dim(str(r)[:2048]).reset()
                    msg_items += [MsgItem(role="tool", tool_call_id=t.id, content=r)]
                    self.message_history += msg_items[-1:]
                    if self.message_queue:
                        msg_items.extend(self.message_queue)
                        self.message_history.extend(self.message_queue)
                        self.message_queue = []
                tool_rounds += 1
            elif resp.finish_reason == "stop":
                final_result = resp.message.content
                break
        self.message_history += [MsgItem(role="assistant", content=final_result)]
        assert isinstance(final_result, str)
        return final_result

    def run(self) -> None:
        os.chdir(self.working_dir)
        super().run()

    def run_shell_tool(
        self,
        command: str,
        stdin: str = "",
        env: dict[str, str] | None = None,
        max_output: int = 16384,
        return_json: bool = False,
    ) -> dict[str, str | int | bool | None] | MsgContent:
        result = self.subshell_helper(["/bin/sh", "-c", command], cwd=self.working_dir, env=env)

        for s in ("stdout", "stderr"):
            r = str(result.get(s, ""))
            if len(r) > max_output:
                result[s] = f"{r[:max_output]}[TRUNCATED]"
                result[f"{s}_truncated"] = True
                result[f"{s}_orig_size"] = len(r)
                result[f"{s}_orig_lines"] = r.count("\n")

        if not return_json:
            return MsgContent(type="text", text=f"{result["stdout"]}{result["stderr"]}")

        return result

    @property
    def shell_tool(self) -> ToolDef:
        desc = "Execute commands in a POSIX shell and return the output.\n"
        "Try to avoid using this to read or write entire files; "
        "use the `read` or `write` tools instead.\n"
        "Each call of this tool will spawn a new subshell."

        return ToolDef(
            func=self.__class__.run_shell_tool,
            meta=ToolFunc(
                name="shell",
                description=desc,
                parameters=ToolParams(
                    properties={
                        "command": ToolProp(
                            "string",
                            "The command string to pass to the shell. "
                            "Compound commands are allowed, e.g. "
                            "\"for fn in $(find . -name '*.txt') ; do echo $(basename $fn) $(grep -o foo | wc -l) ; done\"",
                        ),
                        "return_json": ToolProp(
                            "boolean",
                            r"Whether to return results as structured JSON"
                            r' (e.g. {"stdout": "Hello\n", "stderr": "ERR\n", "returncode": 1})'
                            r' instead of a flat stream of data (e.g. "Hello\nERR\n").',
                        ),
                        "stdin": ToolProp(
                            "string",
                            "A string to use as standard input for the command.",
                        ),
                        "env": ToolProp(
                            "object",
                            "A dictionary of environment variables to add to the environment.",
                        ),
                        "max_output": ToolProp(
                            "integer",
                            "The maximum amount of output to allow before truncating the result.\nDefault: 16384 characters",
                        ),
                    },
                    required=["command"],
                ),
            ),
        )

    def run_web_tool(
        self,
        url: str,
        # data: str = "",
        max_output: int = 16384,
        return_html: bool = False,
    ) -> dict[str, str | int | bool | None] | MsgContent:
        command = (
            f"curl -sSL -A '{self.user_agent} (curl)' -H 'Accept: text/html,text/*;q=0.9' '{url}'"
        )
        if not return_html:
            command = f"lynx -useragent='{self.user_agent} (Lynx)' -dump -dont_wrap_pre -hiddenlinks=ignore -underscore '{url}'"
        result = self.subshell_helper(["/bin/sh", "-c", command], cwd=self.working_dir)

        for s in ("stdout", "stderr"):
            r = str(result.get(s, ""))
            if len(r) > max_output:
                result[s] = f"{r[:max_output]}[TRUNCATED]"
                result[f"{s}_truncated"] = True
                result[f"{s}_orig_size"] = len(r)
                result[f"{s}_orig_lines"] = r.count("\n")

        return MsgContent(type="text", text=f"{result["stdout"]}{result["stderr"]}")

    @property
    def web_tool(self) -> ToolDef:
        desc = "Make simple web requests.\n"
        "By default, HTML will be converted to plain text:"
        " links will have numbers next to them, and"
        " the link targets will be listed at the end.\n"
        "Note: long output will get truncated."

        return ToolDef(
            func=self.__class__.run_web_tool,
            meta=ToolFunc(
                name="web",
                description=desc,
                parameters=ToolParams(
                    properties={
                        "url": ToolProp("string", "The URL to fetch."),
                        "return_html": ToolProp(
                            "boolean",
                            r"Whether to return the original response (typically HTML) or attempt to convert it to plain text.",
                        ),
                        "max_output": ToolProp(
                            "integer",
                            "The maximum amount of output to allow before truncating the result.\nDefault: 16384 characters",
                        ),
                    },
                    required=["url"],
                ),
            ),
        )

    def run_read_tool(self, path: str) -> dict[str, str | int | None] | MsgContent:
        rpath = (self.working_dir / path).resolve()
        try:
            rpath.relative_to(self.working_dir)
        except ValueError:
            self.console.bright("WARNING: reading file outside of working directory!")

        t, _ = mimetypes.guess_file_type(rpath)
        error = None
        status = "No file data read."
        content: MsgContent
        try:
            with rpath.open("rb") as fh:
                fdata = fh.read()
        except (IOError, ValueError) as e:
            error = f"Error reading {path!r}: {e}"

        else:
            status = "File data read successfully."

            try:
                if t in ("image/jpeg", "image/png", "image/gif") and self.has_mmproj:
                    if t == "image/jpeg":
                        # Fix orientation
                        with Image.open(rpath) as im:
                            buf = io.BytesIO()
                            ImageOps.exif_transpose(im, in_place=True)
                            im.save(buf, format="JPEG", quality=90)
                            fdata = buf.getvalue()
                    content = MsgContent(
                        type="image_url",
                        image_url=Url(url=f"data:{t};base64,{b64encode(fdata).decode()}"),
                    )
                else:
                    content = MsgContent(type="text", text=fdata.decode(errors="replace"))
                    if len(str(content.text)) > 65536:
                        error = f"File too big: {len(str(content.text))} chars"
                        status = f"File read but not sent."
            except (IOError, ValueError) as e:
                error = f"Error processing data in {path!r}: {e}"

        if error:
            return {
                "error": error,
                "status": status,
            }
        else:
            return content

    @property
    def read_tool(self) -> ToolDef:
        desc = "Read the contents of the file at `path` into the next tool message."
        if self.has_mmproj:
            desc += " The file can be text or an image."
        return ToolDef(
            func=lambda s, path: self.__class__.run_read_tool(s, path),
            meta=ToolFunc(
                name="read",
                description=desc,
                parameters=ToolParams(
                    properties={
                        "path": ToolProp("string", "The file to read."),
                    },
                    required=["path"],
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

        error = None
        bytes_written = 0
        try:
            with rpath.open("w") as fh:
                bytes_written = fh.write(content)
        except (IOError, ValueError) as e:
            error = f"Couldn't write to {path}: {e}"

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
                description="Write `content` into the file at `path`, "
                "replacing anything that was already there.",
                parameters=ToolParams(
                    properties={
                        "path": ToolProp("string", "The file to write."),
                        "content": ToolProp("string", "The data to write into the file."),
                    },
                    required=["path", "content"],
                ),
            ),
        )

    # TODO: clean this mess up
    @staticmethod
    def updates_from_patch(working_dir: Path, patch: str) -> dict[tuple[Path, Path], str]:
        _patch_lines = iter(patch.splitlines(keepends=True))

        fake_plines: list[str] = []

        def next_pline() -> str:
            if fake_plines:
                return fake_plines.pop(0)
            real_pline = next(_patch_lines, "")
            if real_pline[:1] == "*":
                # Hack in pseudo-patch support
                rpl = real_pline.lower()
                if rpl.startswith("*** update file: "):
                    _, spfn = real_pline.split(":", 1)
                    fake_plines.append(f"+++{spfn}")
                    return f"---{spfn}"

                if rpl.startswith("*** delete file: "):
                    _, spfn = real_pline.split(":", 1)
                    from_path = (working_dir / spfn[1:-1]).resolve()
                    if not from_path.exists():
                        raise ValueError(f"Missing file: {from_path}")

                    fake_plines.append("+++ /dev/null\n")
                    fake_plines.extend(from_path.read_text().splitlines(keepends=True))
                    return f"---{spfn}"

                if rpl.startswith("*** end patch"):
                    return ""

            return real_pline

        patch_line = next_pline()
        updates: dict[tuple[Path, Path], str] = {}
        while patch_line:
            while patch_line:
                patch_prefix = patch_line[:2]
                if patch_prefix == "--":
                    # Found a file
                    break
                if patch_prefix == "@@":
                    raise ValueError(f"Hunk header appears before file name: {patch_line!r}")
                if patch_prefix == "++":
                    raise ValueError(f"Bad file name ordering: {patch_line!r}")
                patch_line = next_pline()

            # File section
            if patch_line[:4] != "--- ":
                raise ValueError(f"Bad file header: {patch_line!r}")
            from_file = patch_line[4:-1]
            if from_file[0] in " \t\n":
                raise ValueError(f"Bad file name: {from_file!r}")
            if from_file[0] == '"':
                from_file = json.loads(from_file)

            patch_line = next_pline()
            if patch_line[:4] != "+++ ":
                raise ValueError(f"Missing file header; got: {patch_line!r}")
            to_file = patch_line[4:-1]
            if to_file[0] in " \t\n":
                raise ValueError(f"Bad file name: {to_file!r}")
            if to_file[0] == '"':
                to_file = json.loads(to_file)

            # TODO: more careful checks?
            if from_file[:2] == "a/":
                from_file = from_file[2:]
            if to_file[:2] == "b/":
                to_file = to_file[2:]

            from_path = (working_dir / from_file).resolve()
            if not from_path.exists():
                raise ValueError(f"File not found: {from_file!r}")

            to_path = (working_dir / to_file).resolve()

            def fline_gen() -> Generator[str, None, None]:
                with from_path.open("r") as from_fh:
                    yield from from_fh.readlines()

            _from_gen = fline_gen()
            from_lines: list[str] = []
            behind = 0

            def next_fline() -> str:
                nonlocal behind
                if behind:
                    line = from_lines[-behind]
                    behind -= 1
                    return line

                line = next(_from_gen, "")
                if line:
                    from_lines.append(line)
                return line

            to_lines: list[str] = []

            patch_line = next_pline()
            from_line = next_fline()
            while patch_line:
                # Hunk header
                m = re.match(r"^@@(?: -?(\d+)(?:,(\d+))? \+?(\d+)(?:,(\d+))?)?", patch_line)
                if m is None:
                    if patch_line[:4] == "--- ":
                        break
                    raise ValueError(f"Missing hunk header; got: {patch_line!r}")
                from_start, from_count, to_start, to_count = (
                    int(g) if g else None for g in m.groups()
                )
                if from_start is None and from_file == "/dev/null":
                    from_start = from_count = 0

                patch_line = next_pline()

                if from_start is not None:
                    if not from_start and from_lines:
                        raise ValueError(
                            "Hunk header says from file should be empty, but it's not"
                        )
                    if from_start < len(from_lines):
                        # Overlapping hunks; reset to/from
                        behind = len(from_lines) - from_start + 1
                        to_lines = to_lines[:behind]
                        from_line = next_fline()
                    else:
                        # Skip to location
                        while len(from_lines) != from_start:
                            from_line = next_fline()
                            if not from_line:
                                raise ValueError("Hunk goes past end of from file")
                            to_lines.append(from_line)
                        if to_start:
                            pass  # TODO: check location?
                else:
                    # Search for hunk
                    if patch_line[0] not in [" ", "-"]:
                        raise ValueError("Need from context to search for unanchored hunks")
                    while True:
                        if from_line == patch_line[1:]:
                            break
                        to_lines.append(from_line)
                        from_line = next_fline()
                        if not from_line:
                            raise ValueError(f"No match in file for {patch_line!r}")

                # Apply hunk
                while patch_line:
                    if from_count is not None and to_count is not None:
                        assert from_start is not None and to_start is not None
                        if len(from_lines) == (from_start + from_count - 1) and len(to_lines) == (
                            to_start + to_count - 1
                        ):
                            break

                    line_prefix, line_content = patch_line[:1], patch_line[1:]
                    match line_prefix:
                        case " ":
                            if line_content != from_line:
                                raise ValueError(
                                    f"Context mismatch; expected {line_content!r} but got {from_line!r}"
                                )
                            to_lines.append(line_content)
                            patch_line = next_pline()
                            from_line = next_fline()
                        case "-":
                            if from_count is None and patch_line[:4] == "--- ":
                                break  # Assume new file
                            if line_content != from_line:
                                raise ValueError(
                                    f"Line deletion mismatch: {line_content!r} != {from_line!r}"
                                )
                            patch_line = next_pline()
                            from_line = next_fline()
                        case "+":
                            to_lines.append(line_content)
                            patch_line = next_pline()
                        case "@":
                            # TODO if (...): raise ValueError("Bad line counts")
                            break
                        case "":
                            break
                        case _:
                            raise ValueError(f"Bad line mid-hunk: {patch_line!r}")

            while from_line:
                to_lines.append(from_line)
                from_line = next_fline()

            updates[(from_path, to_path)] = "".join(to_lines)

        return updates

    def run_patch_tool(self, patch: str, path: str = "") -> dict[str, str | int | None]:
        rpath = (self.working_dir / path).resolve()
        try:
            rpath.relative_to(self.working_dir)
        except ValueError:
            return {
                "error": f"{path!r} not in working directory",
                "files_updated": 0,
            }

        try:
            updates = self.updates_from_patch(rpath, patch)
        except ValueError as e:
            return {
                "error": f"Could not apply patch: {e}",
                "files_updated": 0,
            }
        if not updates:
            return {
                "error": "No file updates found in unified diff format",
                "files_updated": 0,
            }

        error = None
        files_updated = 0
        devnull = Path("/dev/null")
        try:
            # TODO: safer updates?
            for (from_path, to_path), data in updates.items():
                to_path.write_text(data)
                files_updated += 1
                if from_path not in (to_path, devnull):
                    from_path.unlink()
        except OSError as e:
            error = str(e)

        return {
            "error": error,
            "files_updated": files_updated,
        }

    @property
    def patch_tool(self) -> ToolDef:
        return ToolDef(
            func=self.__class__.run_patch_tool,
            meta=ToolFunc(
                name="apply_patch",
                description="Apply the given patch relative the given directory.",
                parameters=ToolParams(
                    properties={
                        "patch": ToolProp(
                            "string",
                            "The patch (in unified diff format) to apply.\n"
                            "Be sure to include trailing newlines if necessary.\n"
                            """For example, to rename `foo.txt` to `bar.txt` and modify the second line while also creating baz.txt:
--- foo.txt
+++ bar.txt
@@ -1,3 1,3 @@
 a
-b
+B
 c
--- /dev/null
+++ baz.txt
@@ -0,0 +1,1 @@
+D
""",
                        ),
                        "path": ToolProp(
                            "string",
                            "The path that files will be made relative to.\n"
                            "Defaults to the workspace directory.",
                        ),
                    },
                    required=["patch"],
                ),
            ),
        )


if __name__ == "__main__":
    import tomllib

    match len(sys.argv):
        case 1:
            conf_id = "Agent"
        case 2:
            conf_id = sys.argv[1]
        case _:
            raise ValueError("Too many arguments")

    with Path("./agent.toml").open("rb") as fh:
        agent_config = tomllib.load(fh)[conf_id]

    client = Client("http://host.docker.internal:8080/v1")
    console = Console(history_file=".agent_history")

    agent = ToolAgent(
        client=client,
        console=console,
        **agent_config,
    )

    try:
        agent.run()
    except EOFError:
        pass
