import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Incoming:
    id: int
    date: datetime
    message: str | None


@dataclass(frozen=True)
class Route:
    source: object
    destination: object
    discussion: object
    bot_discussion: object


class Undeliverable(Exception):
    pass


class ForwardRestricted(Exception):
    pass


async def index_batch(archive, telegram, source, batch_size=40, should_continue=lambda: True):
    state = archive.state(source.channel_id)
    head = state['snapshot']
    if head is None:
        head = max(state['cursor'], await telegram.head(source))
        if state['history_done'] and head == state['cursor']:
            return False
        archive.snapshot(source.channel_id, head)
    count = 0
    if head > state['cursor']:
        async for message in telegram.history(source, state['cursor'], head, batch_size):
            if not should_continue():
                return True
            archive.index(source.channel_id, message.id, message.date.isoformat(), message.message)
            count += 1
            await asyncio.sleep(0)
    if not should_continue():
        return True
    if count < batch_size or archive.state(source.channel_id)['cursor'] >= head:
        archive.finish_snapshot(source.channel_id, head)
        if not state['history_done']:
            LOG.info('История канала %s проиндексирована до сообщения %s', source.channel_id, head)
        return False
    return True


async def deliver(archive, telegram, route):
    source_id = route.source.channel_id
    job = archive.next_job(source_id)
    if job is None:
        return False
    post = job['id']
    forward = job['forward_id']
    try:
        if forward is None:
            if not job['copy_required']:
                try:
                    forward = await telegram.forward(route, post, job['forward_random'])
                except ForwardRestricted:
                    archive.update_delivery(source_id, post, copy_required=True)
            if forward is None:
                forward = await telegram.copy(route, post, job['forward_random'])
            archive.update_delivery(source_id, post, forward=forward)
        root = job['discussion_root']
        if root is None:
            root = await telegram.discussion_root(route, forward)
            archive.update_delivery(source_id, post, root=root)
        result = await telegram.send_report(route, root, archive.report_for(job), job['report_random'])
        archive.update_delivery(source_id, post, result=result)
    except Undeliverable as exc:
        LOG.warning('Источник %s, пост %s: публикация результата невозможна (%s)', source_id, post, exc)
        archive.update_delivery(source_id, post, skipped=str(exc))
    return True
