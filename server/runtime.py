from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Thread
from uuid import uuid4
import traceback
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.request import Request
from engine.scheduler import Scheduler
from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache
from engine.continuous_batch_engine import ContinuousBatchEngine


MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda"
BLOCK_SIZE = 16
TOTAL_BLOCKS = 256
MAX_BATCH_SIZE = 32
MAX_CONTEXT_LEN = 512


@dataclass
class GenerationJob:
    prompt: str
    max_new_tokens: int
    request_id: str = field(default_factory=lambda: uuid4().hex)
    future: Future = field(default_factory=Future)
    events: Queue | None = None


class TinyServeRuntime:
    def __init__(self):
        self.queue = Queue(maxsize=128)
        self.stop_event = Event()
        self.ready_event = Event()
        self.failure = None
        self.worker = None

    def start(self):
        self.worker = Thread(target=self._run, daemon=True)
        self.worker.start()
        self.ready_event.wait()

        if self.failure is not None:
            raise RuntimeError("TinyServe initialization failed") from self.failure

    def stop(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join()

    def submit(self, prompt, max_new_tokens):
        if self.failure is not None or self.stop_event.is_set():
            raise RuntimeError("Inference worker is unavailable")

        job = GenerationJob(prompt, max_new_tokens)
        self.queue.put_nowait(job)
        return job.future

    def submit_stream(self, prompt, max_new_tokens):
        if self.failure is not None or self.stop_event.is_set():
            raise RuntimeError("Inference worker is unavailable")

        job = GenerationJob(prompt, max_new_tokens)

        # One output channel per streaming request
        job.events = Queue(maxsize=max_new_tokens + 2)

        self.queue.put_nowait(job)
        return job.request_id, job.events

    def _initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32, attn_implementation="eager"
        ).to(DEVICE).eval()

        config = self.model.config

        manager = KVCacheManager(
            num_blocks=TOTAL_BLOCKS, block_size=BLOCK_SIZE
        )
        pool = KVCachePool(
            num_layers=config.num_hidden_layers,
            num_blocks=TOTAL_BLOCKS,
            num_kv_heads=config.num_key_value_heads,
            block_size=BLOCK_SIZE,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device=DEVICE,
        )
        cache = PagedKVCache(manager, pool)

        scheduler = Scheduler(
            max_batch_size=MAX_BATCH_SIZE,
            total_kv_blocks=TOTAL_BLOCKS,
            block_size=BLOCK_SIZE,
        )

        self.scheduler = scheduler
        self.engine = ContinuousBatchEngine(
            self.model, scheduler, cache, pool, manager,
            eos_token_id=self.tokenizer.eos_token_id,
            attention_backend="cuda_graph",
        )

        print("TinyServe inference worker ready")

    def _run(self):
        active = {}
        stream_events = {}

        try:
            self._initialize()
            self.ready_event.set()

            while not self.stop_event.is_set():
                # Wait when no requests are outstanding
                jobs = []

                if not active:
                    try:
                        jobs.append(self.queue.get(timeout=0.1))
                    except Empty:
                        continue

                # Collect other requests that have arrived
                while True:
                    try:
                        jobs.append(self.queue.get_nowait())
                    except Empty:
                        break

                # Submit new requests to the scheduler
                for job in jobs:
                    if job.future.cancelled():
                        continue

                    try:
                        prompt_ids = self.tokenizer.encode(job.prompt)

                        if not prompt_ids:
                            raise ValueError("Prompt produced no tokens")

                        if len(prompt_ids) + job.max_new_tokens > MAX_CONTEXT_LEN:
                            raise ValueError("Request exceeds maximum context length")

                        req = Request(
                            job.request_id, job.prompt, prompt_ids,
                            max_new_tokens=job.max_new_tokens,
                        )

                        self.scheduler.add_request(req)
                        active[req.request_id] = (req, job.future)

                        if job.events is not None:
                            stream_events[req.request_id] = job.events
                        
                    except Exception as exc:
                        if not job.future.done():
                            job.future.set_exception(exc)

                # Run one continuous-batching iteration
                if active:
                    # Record how many tokens each request had before this tick
                    previous = {
                        rid: len(req.generated_ids)
                        for rid, (req, _) in active.items()
                    }

                    self.engine.step()

                    for rid, (req, future) in list(active.items()):
                        events = stream_events.get(rid)

                        # Stream every token newly produced in this tick
                        if events is not None:
                            for j in range(previous[rid], len(req.generated_ids)):
                                events.put_nowait({
                                    "event": "token",
                                    "request_id": rid,
                                    "token_id": req.generated_ids[j],
                                    "text": self.tokenizer.decode(
                                        req.generated_ids[:j + 1],
                                        skip_special_tokens=True,
                                    ),
                                })

                        if not req.is_finished:
                            continue

                        result = {
                            "request_id": rid,
                            "text": self.tokenizer.decode(
                                req.generated_ids, skip_special_tokens=True
                            ),
                            "generated_token_ids": list(req.generated_ids),
                        }

                        if events is not None:
                            events.put_nowait({
                                "event": "done",
                                "request_id": rid,
                                "text": result["text"],
                            })
                            del stream_events[rid]

                        if not future.done():
                            future.set_result(result)

                        del active[rid]

        except Exception as exc:
            self.failure = exc
            traceback.print_exc()
            for req, future in active.values():
                if not future.done():
                    future.set_exception(exc)
        
        finally:
            self.ready_event.set()

            # 1. Handle jobs still waiting in the HTTP queue
            while True:
                try:
                    job = self.queue.get_nowait()
                except Empty:
                    break

                if not job.future.done():
                    job.future.set_exception(
                        RuntimeError("Inference worker stopped")
                    )

                # Notify waiting streaming clients
                if job.events is not None:
                    job.events.put_nowait({
                        "event": "error",
                        "request_id": job.request_id,
                        "detail": "Inference worker stopped",
                    })

            # 2. Handle requests already admitted by the scheduler
            for req, future in active.values():
                if not future.done():
                    future.set_exception(
                        RuntimeError("Inference worker stopped")
                    )

            # 3. Notify active streaming clients
            for rid, events in stream_events.items():
                events.put_nowait({
                    "event": "error",
                    "request_id": rid,
                    "detail": "Inference worker stopped",
                })
