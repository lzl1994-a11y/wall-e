"""Control contract for the long-running RDK tracking vision pipeline."""


VISION_PIPELINE_COMMAND_TOPIC = "/vision_pipeline_cmd"
VISION_PIPELINE_START = "start"
VISION_PIPELINE_STOP = "stop"


def decode_vision_pipeline_command(raw: str) -> str | None:
    command = str(raw or "").strip().lower()
    if command in {VISION_PIPELINE_START, VISION_PIPELINE_STOP}:
        return command
    return None
