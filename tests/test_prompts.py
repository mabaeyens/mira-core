

def test_current_datetime_states_the_offset_and_the_utc_time():
    """The stamp once read just "19:49 CET"; the model took it as UTC and answered 20:49
    to "what time is it in UTC+1". It must carry the numeric offset and the UTC time."""
    from datetime import datetime, timedelta, timezone
    from core.prompts import current_datetime_str
    cet = timezone(timedelta(hours=1), "CET")
    s = current_datetime_str(datetime(2026, 11, 21, 19, 49, tzinfo=cet))
    assert s == "November 21, 2026, 19:49 CET (UTC+01:00; UTC time 18:49)"
    ist = timezone(timedelta(hours=5, minutes=30), "IST")
    assert "UTC+05:30; UTC time 14:19" in current_datetime_str(datetime(2026, 1, 1, 19, 49, tzinfo=ist))
    assert "UTC-05:00; UTC time 00:49" in current_datetime_str(
        datetime(2026, 1, 1, 19, 49, tzinfo=timezone(timedelta(hours=-5), "EST")))
