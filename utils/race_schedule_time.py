"""Keep the source racing day separate from its absolute scheduled jump."""
from datetime import datetime, timedelta
import re
from zoneinfo import ZoneInfo

MELBOURNE = ZoneInfo('Australia/Melbourne')


def scheduled_jump_datetime(record):
    """Validate an explicit jump; callers must not fall back after invalid evidence."""
    try:
        jump = datetime.fromisoformat(str(record['scheduled_jump_datetime']))
        if jump.utcoffset() is None:
            return None
        jump = jump.astimezone(MELBOURNE)
        day = datetime.strptime(str(record.get('date') or record.get('race_date')), '%Y-%m-%d').date()
        # A meeting dated in Australia may finish after Melbourne midnight.
        if jump.date() not in (day, day + timedelta(days=1)):
            return None
        return jump
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def extract_formatted_timing(soup):
    """Read a unique official epoch, retaining clock-only compatibility if absent.

    Malformed or conflicting explicit timing is an error, never permission to
    search unrelated page clocks. Epochs define the instant across timezones.
    """
    for selector in ('formatted-time[data-format="datetime_short"]', 'formatted-time[data-format="time_24"]'):
        elements = soup.select(selector)
        if not elements:
            continue
        stamps = [e.get('data-timestamp') for e in elements if e.has_attr('data-timestamp')]
        if stamps:
            if len(stamps) != len(elements) or any(not re.fullmatch(r'[0-9]+', str(s)) for s in stamps) or len(set(stamps)) != 1:
                raise ValueError('official_jump_timestamp_ambiguous_or_invalid')
            try:
                jump = datetime.fromtimestamp(int(stamps[0]), MELBOURNE)
            except (ValueError, OverflowError, OSError) as exc:
                raise ValueError('official_jump_timestamp_invalid') from exc
            return {'race_time': jump.strftime('%I:%M %p').lstrip('0'), 'scheduled_jump_datetime': jump.isoformat()}
        clocks = set()
        for element in elements:
            match = re.search(r'\b(\d{1,2}:\d{2}(?:\s*[AP]M)?)\b', element.get_text(' ', strip=True), re.I)
            if not match:
                raise ValueError('official_jump_clock_invalid')
            clock = match[1].upper().replace(' ', '')
            try:
                value = datetime.strptime(clock, '%I:%M%p' if clock.endswith(('AM','PM')) else '%H:%M')
            except ValueError as exc:
                raise ValueError('official_jump_clock_invalid') from exc
            clocks.add(value.strftime('%I:%M %p').lstrip('0'))
        if len(clocks) != 1:
            raise ValueError('official_jump_clock_ambiguous')
        return {'race_time': clocks.pop()}
    return {}
