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


async def deliver(archive, telegram, route, should_continue=lambda: True):
    source = route.source.channel_id
    job = archive.next_job(source)
    if job is None:
        return False
    post = job['post']
    if not should_continue() or not archive.job_pending(source, post):
        return False
    try:
        forward = job['forward_id']
        if forward is None:
            try:
                forward = await telegram.forward(route, post, job['forward_random'], reuse_copy=False)
            except ForwardRestricted:
                forward = await telegram.copy(route, post, job['forward_random'], reuse_copy=False)
            archive.update_job(source, post, forward=forward)
        if not should_continue() or not archive.job_pending(source, post):
            return False
        original = archive.post_text(source, post)
        result = await telegram.send_report(route, forward, job['result'], job['report_random'], original)
        archive.update_job(source, post, result=result)
    except Undeliverable as exc:
        archive.update_job(source, post, skipped=str(exc))
        archive.remember_error(source, f'Публикация {post}: {exc}')
        LOG.warning('Публикация %s/%s пропущена: %s', source, post, exc)
    return True
