"""Policy readiness Query; benchmark Sessions own start/reset/stop commands."""
from __future__ import annotations
import asyncio
import time
import uuid
from typing import Any, Callable
from forge_msgs import ToolMessage
from forge_tool import (TOOL_ENDPOINT_PROTOCOL, ToolEndpointDescriptor, ToolEndpointHandler,
                        ToolOperationDescriptor, ToolResult)
from forge_tool.dora import DoraToolEndpointBinding, tool_envelope_to_message, tool_message_to_envelope
from bench_tool.lease import EndpointLease


class PolicyDescribe:
    def __init__(self, describe: Callable[[], dict[str, Any]]):
        self.describe = describe

    async def query(self, request: Any, context: Any) -> ToolResult:
        return ToolResult(status="succeeded", outputs=self.describe())


class PolicyTool:
    """Construct only after model initialization/WS connection has succeeded."""
    def __init__(self, node: Any, describe: Callable[[], dict[str, Any]]):
        self.node = node
        self.loop = asyncio.Runner()
        descriptor = ToolEndpointDescriptor(
            protocol_version=TOOL_ENDPOINT_PROTOCOL, endpoint_id="policy.runner",
            operations=(ToolOperationDescriptor(name="describe", semantics="query", max_concurrency=1),))
        instance = str(uuid.uuid4())
        handler = ToolEndpointHandler(descriptor, endpoint_instance_id=instance,
                                      operations={"describe": PolicyDescribe(describe)})
        self.binding = DoraToolEndpointBinding(handler, event_sink=self._send_batch)
        self.lease = EndpointLease(descriptor, instance)

    async def _send_batch(self, batch: Any) -> None:
        self.node.send_output("tool_out", batch)

    def _send(self, envelope: Any) -> None:
        if envelope is not None:
            self.node.send_output("tool_out", tool_envelope_to_message(envelope).to_arrow())

    def tick(self) -> None:
        self._send(self.lease.poll(time.monotonic()))

    def process(self, value: Any) -> None:
        envelope = tool_message_to_envelope(ToolMessage.from_arrow(value))
        if envelope.message_type == "endpoint.registry.response":
            self.lease.acknowledge(envelope, time.monotonic())
        else:
            for batch in self.loop.run(self.binding.dispatch_input(value)):
                self.node.send_output("tool_out", batch)

    def close(self) -> None:
        try:
            self._send(self.lease.stop())
        finally:
            self.loop.close()
