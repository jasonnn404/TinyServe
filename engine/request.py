from enum import Enum, auto


class RequestStatus(Enum):
    PREFILL = auto()
    DECODE = auto()
    FINISHED = auto()


class Request:
    def __init__(self, request_id, prompt, prompt_ids, max_new_tokens=100):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_ids = prompt_ids
        self.generated_ids = []
        self.max_new_tokens = max_new_tokens
        self.status = RequestStatus.PREFILL

    def mark_prefilled(self):
        if self.status != RequestStatus.PREFILL:
            raise ValueError("Request is not waiting for prefill")

        self.status = RequestStatus.DECODE

    def add_token(self, token_id, eos_token_id):
        self.generated_ids.append(token_id)

        if token_id == eos_token_id:
            self.status = RequestStatus.FINISHED
        elif len(self.generated_ids) >= self.max_new_tokens:
            self.status = RequestStatus.FINISHED
        else:
            self.status = RequestStatus.DECODE

    @property
    def is_finished(self):
        return self.status == RequestStatus.FINISHED

    @property
    def token_ids(self):
        return self.prompt_ids + self.generated_ids