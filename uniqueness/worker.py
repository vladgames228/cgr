import asyncio
import logging
import time
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
    if state['index_limit'] is not None and state['scan_count'] >= state['index_limit']:
        archive.set_route_active(source.channel_id, 2 if state['auto_publish'] else 0)
        return False
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
            state = archive.state(source.channel_id)
            if state['index_limit'] is not None and state['scan_count'] >= state['index_limit']:
                archive.set_route_active(source.channel_id, 2 if state['auto_publish'] else 0)
                return False
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
    snoozed = archive.reply_until(source_id) > time.time()
    job = archive.next_forward_job(source_id) if snoozed else archive.next_job(source_id)
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
        if snoozed:
            return True
        result = await telegram.send_report(route, root, archive.report_for(job), job['report_random'])
        archive.update_delivery(source_id, post, result=result)
    except Undeliverable as exc:
        LOG.warning('Источник %s, пост %s: публикация результата невозможна (%s)', source_id, post, exc)
        archive.update_delivery(source_id, post, skipped=str(exc))
    return True


async def deliver_manual(archive, telegram, route):
    source = route.source.channel_id
    snoozed = archive.reply_until(source) > time.time()
    job = archive.next_manual(source, forward_only=snoozed)
    if job is None:
        return False
    sender, request_id, post = job['sender'], job['request_id'], job['post']
    try:
        forward = job['forward_id']
        if forward is None:
            try:
                forward = await telegram.forward(route, post, job['forward_random'], reuse_copy=False)
            except ForwardRestricted:
                forward = await telegram.copy(route, post, job['forward_random'], reuse_copy=False)
            archive.update_manual(sender, request_id, forward=forward)
        root = job['discussion_root']
        if root is None:
            root = await telegram.discussion_root(route, forward)
            archive.update_manual(sender, request_id, root=root)
        if snoozed:
            return True
        result = await telegram.send_report(route, root, job['result'], job['report_random'])
        archive.update_manual(sender, request_id, result=result)
    except Undeliverable as exc:
        archive.update_manual(sender, request_id, skipped=str(exc))
        LOG.warning('Ручная публикация %s/%s пропущена: %s', sender, request_id, exc)
    return True
