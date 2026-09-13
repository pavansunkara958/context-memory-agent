"""The knowledge base: Helios field-service documentation.

Same fictional product as Modules 4 and 6 — an industrial IoT platform with
HX-40 gateways and TS-9 probes deployed at customer sites.

The corpus is written to make retrieval *measurable*, not merely plausible:

  - Documents disagree across versions. REL-5.0 changes behaviour that RB-105
    describes, so the right answer depends on which version the engineer runs.
  - Some documents overlap heavily by design (RB-107 restates the rollback
    procedure from REL-5.0) so near-duplicate removal has something real to do.
  - Several documents are lexical distractors: they contain the query's
    keywords but answer a different question. Dense retrieval should beat BM25
    on those, and the ablation table shows whether it does.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    kind: str          # runbook | release | incident | api | hardware | policy
    date: str
    text: str


DOCUMENTS: list[Document] = [
    # ---------------------------------------------------------------- runbooks
    Document(
        "RB-101", "Replacing a failed HX-40 gateway radio", "runbook", "2026-01-14",
        """Symptoms: the gateway powers on, the status LED is solid amber, and no
        uplinks reach the platform for more than fifteen minutes.

        Before replacing anything, confirm the radio is genuinely dead. Run
        `helios-cli radio selftest` on the unit. Exit code 3 means the radio
        module failed its internal loopback and the board must be swapped.
        Exit code 0 with no uplinks points at antenna or backhaul, not the radio.

        To replace: power down, remove the four M3 screws on the underside,
        disconnect the u.FL pigtail gently — it tears if pulled sideways — and
        lift the radio daughterboard straight up. Fit the replacement, torque the
        screws to 0.4 Nm, and re-pair with `helios-cli radio pair --site <id>`.

        Record the old module serial in the RMA ticket. A radio swap always
        requires an RMA under the hardware warranty policy."""),

    Document(
        "RB-102", "Recovering a node stuck in bootloader", "runbook", "2026-02-03",
        """A node that reboots into the bootloader repeatedly shows a rapidly
        blinking green LED, roughly four flashes per second, and never joins the
        network.

        This almost always follows an interrupted firmware write. Connect over
        USB-C, run `helios-cli node console`, and look for `CRC mismatch at
        slot B`. That confirms a partial image.

        Recovery: hold the RESET button for ten seconds to force slot A, which
        holds the last known-good image. The node should rejoin within ninety
        seconds. Then re-flash slot B with `helios-cli node flash --slot b`.

        Do not raise an RMA for a bootloader loop. The hardware is fine; this is
        recoverable in the field in under ten minutes."""),

    Document(
        "RB-103", "Calibrating the TS-9 temperature probe", "runbook", "2026-01-28",
        """The TS-9 ships factory-calibrated but drifts roughly 0.1 degrees Celsius
        per year of continuous operation, and faster above 60 degrees ambient.

        Calibrate against a reference bath at two points, 0 and 50 degrees.
        Let the probe settle for a full five minutes at each point; readings taken
        earlier are dominated by thermal mass and will bake an error into the
        curve.

        Submit the two readings with `helios-cli probe calibrate --low <v>
        --high <v>`. The platform computes a linear correction and pushes it to
        the device on the next uplink.

        Recalibrate annually, or immediately after any reading that disagrees with
        a neighbouring probe by more than 2 degrees."""),

    Document(
        "RB-104", "Rotating field device certificates", "runbook", "2026-03-11",
        """Device certificates are valid for 24 months. The platform emits a
        `cert.expiring` webhook at 30, 14 and 3 days before expiry.

        Rotation is online and needs no site visit. Run `helios-cli fleet
        rotate-certs --site <id>`. Devices pick up the new certificate on their
        next check-in and switch over at the following reconnect, so a full site
        completes within one reporting interval.

        A device that has already expired cannot rotate over the air — it can no
        longer establish the TLS session needed to fetch the new certificate.
        Those units require a USB-C re-provision on site, which is why the 30-day
        warning exists and why ignoring it is expensive."""),

    Document(
        "RB-105", "Clearing a full local buffer on HX-40", "runbook", "2026-02-19",
        """Each HX-40 buffers telemetry locally when backhaul is unavailable. The
        buffer holds 72 hours at the default one-minute sampling interval.

        When the buffer fills, firmware 4.x DISCARDS THE OLDEST records to make
        room for new ones. Check occupancy with `helios-cli buffer status`.

        To drain a full buffer once backhaul returns, run `helios-cli buffer
        flush --rate 200`. The rate limit matters: flushing at full speed from
        several gateways at once has previously saturated the ingest tier.

        Note this behaviour changed in firmware 5.0. Confirm the firmware version
        before relying on the discard-oldest assumption."""),

    Document(
        "RB-106", "Diagnosing intermittent LoRa packet loss", "runbook", "2026-03-02",
        """Intermittent loss — uplinks arriving but with gaps — is usually RF, not
        software.

        Collect `helios-cli radio stats --window 24h` and read three numbers.
        RSSI below -110 dBm means the link is marginal. SNR below -7 dB means the
        signal is under the noise floor. A packet error rate above 5 per cent with
        healthy RSSI and SNR points at interference rather than distance.

        Check antenna placement first: an antenna mounted flat against metal loses
        roughly 12 dB, which is more than most site surveys account for. Raising
        it 30 centimetres off the surface usually recovers the link.

        Only escalate to hardware once placement and interference are excluded."""),

    Document(
        "RB-107", "Rolling back HX-40 firmware", "runbook", "2026-04-08",
        """Rollback is supported from 5.x to 4.3 only. Earlier targets are blocked
        because the on-device schema changed and 4.2 cannot read a 5.x buffer.

        Run `helios-cli fleet rollback --to 4.3 --site <id>`. The fleet rolls in
        batches of ten with a five-minute soak between batches, so a hundred-device
        site takes about an hour.

        Buffered telemetry written under 5.x is NOT readable after rolling back to
        4.3. Flush the buffer first with `helios-cli buffer flush` or accept the
        loss. This is the single most common mistake during a rollback and it is
        not recoverable afterwards."""),

    # ---------------------------------------------------------------- releases
    Document(
        "REL-4.2", "Helios firmware 4.2 release notes", "release", "2025-09-30",
        """Adds support for the TS-9 Rev C probe. Reduces idle current draw by
        18 per cent, extending battery life on solar-assisted sites.

        Fixes a bug where `helios-cli buffer status` reported occupancy as a
        percentage of 64 hours rather than the actual 72-hour capacity.

        Known issue: certificate rotation can fail silently on devices with clock
        drift above five minutes. Sync time before rotating."""),

    Document(
        "REL-4.3", "Helios firmware 4.3 release notes", "release", "2025-12-12",
        """Fixes the silent certificate-rotation failure from 4.2 by forcing an NTP
        sync before the rotation handshake.

        Adds `helios-cli radio selftest`, which exercises the radio's internal
        loopback and returns exit code 3 on a hardware fault. This replaces the
        older visual LED diagnosis, which produced a lot of unnecessary RMAs.

        4.3 is the oldest release supported as a rollback target from 5.x."""),

    Document(
        "REL-5.0", "Helios firmware 5.0 release notes", "release", "2026-02-26",
        """BREAKING: the local buffer now applies BACKPRESSURE instead of
        discarding the oldest records. When the buffer fills, the device reduces
        its sampling rate rather than losing history. Dashboards that assumed a
        continuous one-minute series must handle variable intervals.

        BREAKING: `/v1/telemetry` responses are cursor-paginated. Offset
        pagination is removed, not deprecated.

        The on-device buffer schema changed. Telemetry written under 5.0 cannot
        be read by 4.x firmware, so a rollback loses any unflushed buffer.

        Rollback to 4.3 is supported. Rollback to 4.2 or earlier is blocked."""),

    Document(
        "REL-5.1", "Helios firmware 5.1 release notes", "release", "2026-04-30",
        """Adds a `buffer.backpressure` webhook so backpressure events are visible
        without polling.

        Reduces rollback batch soak time from ten minutes to five, roughly halving
        the time to roll a large site.

        Fixes a regression in 5.0 where `helios-cli buffer flush --rate` ignored
        the rate argument and always flushed at full speed."""),

    # --------------------------------------------------------------- incidents
    Document(
        "INC-2291", "Postmortem: fleet-wide telemetry gap", "incident", "2026-01-09",
        """Impact: 4,100 devices stopped reporting for 6 hours 40 minutes.

        Cause: an ingest deploy tightened payload validation to reject unknown
        fields. Devices on 4.2 send a `debug_flags` field that 4.3 removed. The
        stricter validator rejected every 4.2 uplink.

        Detection took 51 minutes because the ingest dashboard tracks accepted
        requests per second and the rejections were counted as handled. Volume
        looked normal. The gap was visible only in per-device last-seen.

        Actions: validators now warn on unknown fields for one release before
        rejecting, and the ingest dashboard alerts on last-seen distribution
        rather than request volume."""),

    Document(
        "INC-2314", "Postmortem: certificate expiry outage", "incident", "2026-02-14",
        """Impact: 312 devices at two sites went offline and required on-site
        re-provisioning. Recovery took nine days of field visits.

        Cause: the `cert.expiring` webhook was configured against a decommissioned
        endpoint. All 30, 14 and 3-day warnings were delivered successfully to a
        URL nobody read. Certificates expired, and expired devices cannot rotate
        over the air because they can no longer complete the TLS handshake.

        The webhook was returning HTTP 200 from a default handler, so delivery
        monitoring showed healthy. A successful delivery is not a received
        warning.

        Actions: quarterly webhook endpoint verification, and expiry warnings now
        also raise a ticket rather than relying on a webhook alone."""),

    Document(
        "INC-2350", "Postmortem: export timeouts above 500k rows", "incident",
        "2026-03-21",
        """Impact: CSV exports failed for any dataset above roughly 500,000 rows.
        Around 40 customers affected over eleven days.

        Cause: the export path materialised the full result set in memory before
        writing. Above 500k rows the worker exceeded its memory limit and was
        killed, surfacing to the customer as a gateway timeout with no
        explanation.

        Workaround issued at the time: filter by date range to keep exports under
        the threshold. Fixed by streaming the result set to object storage and
        emailing a signed link.

        This is known issue KI-118 in customer-facing documentation."""),

    Document(
        "INC-2377", "Postmortem: duplicate telemetry rows after flush", "incident",
        "2026-04-17",
        """Impact: roughly 2.1 million duplicate telemetry rows across 180 devices
        over three days.

        Cause: `helios-cli buffer flush` did not mark records as acknowledged
        until the entire batch completed. A backhaul drop mid-batch caused the
        device to retry from the start of the batch, resending records the
        platform had already accepted.

        Ingest deduplicates on (device_id, timestamp), but the 5.0 backpressure
        change means timestamps are no longer evenly spaced, and the dedup window
        assumed a fixed one-minute grid. Records that fell outside the window were
        stored twice.

        Actions: per-record acknowledgement during flush, and the dedup window is
        now derived from the device's reported interval."""),

    # -------------------------------------------------------------------- api
    Document(
        "API-001", "API authentication", "api", "2026-03-01",
        """All requests authenticate with a bearer token in the `Authorization`
        header. Tokens are scoped per site and per role.

        Tokens do not expire but can be revoked. Revocation propagates within 60
        seconds. There is no refresh flow; mint a new token and swap it in.

        Never send a token as a query parameter. The gateway logs full request
        URLs, so a token in a query string ends up in log storage and in any
        downstream log aggregation."""),

    Document(
        "API-002", "GET /v1/devices", "api", "2026-03-01",
        """Lists devices for a site. Query parameters: `site_id` (required),
        `status` (one of online, offline, degraded), `firmware`, `cursor`,
        `limit` (default 50, maximum 200).

        Returns `device_id`, `serial`, `firmware`, `last_seen`, `status`,
        `cert_expires_at`.

        `last_seen` is the last accepted uplink, not the last attempted one. A
        device whose uplinks are being rejected by validation will show a stale
        `last_seen` while appearing healthy in request-volume metrics."""),

    Document(
        "API-003", "GET /v1/telemetry", "api", "2026-03-01",
        """Returns time-series readings. Parameters: `device_id` or `site_id`,
        `from`, `to` (ISO 8601), `interval`, `cursor`, `limit` (maximum 1000).

        As of firmware 5.0 and the matching platform release, this endpoint is
        cursor-paginated only. Offset pagination has been removed. Pass the
        `next_cursor` from the previous response.

        Readings are returned in ascending timestamp order. Intervals are not
        guaranteed uniform: a device under backpressure reduces its sampling
        rate, so consumers must not assume a fixed grid."""),

    Document(
        "API-004", "Rate limits", "api", "2026-03-01",
        """Default limits are 60 requests per minute per token for read endpoints
        and 10 per minute for write endpoints. Exports count as writes.

        Exceeding a limit returns HTTP 429 with a `Retry-After` header in
        seconds. Clients should honour it; retrying immediately extends the
        penalty window.

        Bulk reads should use a larger `limit` rather than more requests. One
        request at limit 1000 costs the same against the quota as one request at
        limit 50."""),

    Document(
        "API-005", "Webhooks", "api", "2026-03-01",
        """Subscribe to `device.offline`, `cert.expiring`, `buffer.full`,
        `buffer.backpressure` and `export.ready`.

        Deliveries retry with exponential backoff for 24 hours. Any 2xx response
        counts as delivered — the platform cannot distinguish a handler that
        processed the event from a default route that returned 200 and discarded
        it.

        Verify the `X-Helios-Signature` HMAC on every delivery. Verify your
        endpoint quarterly; a silently-discarding endpoint is indistinguishable
        from a working one in delivery metrics."""),

    Document(
        "API-006", "Pagination", "api", "2026-03-01",
        """Cursor pagination throughout. Responses include `next_cursor` when more
        data exists and omit it on the final page.

        Cursors are opaque and expire after one hour. Do not construct, parse or
        store them.

        Offset pagination was removed alongside firmware 5.0. Code that passes
        `offset` receives HTTP 400."""),

    # --------------------------------------------------------------- hardware
    Document(
        "HW-01", "HX-40 gateway specification", "hardware", "2025-11-05",
        """LoRa gateway. Operating range -30 to +65 degrees Celsius. IP66. Powered
        by 12V DC or PoE. Idle draw 2.1W, peak 6.4W during flush.

        Local storage 8 GB, which is 72 hours of telemetry at the default
        one-minute interval across 200 attached nodes.

        Radio module is a field-replaceable daughterboard. The u.FL pigtail is the
        most commonly damaged part during service and is not sold separately —
        a damaged pigtail means replacing the whole daughterboard."""),

    Document(
        "HW-02", "TS-9 probe specification", "hardware", "2025-11-05",
        """Temperature probe. Range -40 to +125 degrees Celsius, accuracy plus or
        minus 0.3 degrees after calibration. Drift approximately 0.1 degrees per
        year.

        Battery: 3.6V lithium thionyl chloride, typical life five years at a
        one-minute sampling interval, dropping to under two years at ten-second
        sampling.

        Rev C adds a shielded cable and requires firmware 4.2 or later. Rev B
        probes on 5.x firmware report correctly but cannot use the faster
        sampling modes."""),

    Document(
        "HW-03", "Power and battery guidance", "hardware", "2026-01-20",
        """Battery life scales roughly with the square root of the sampling
        interval, not linearly — halving the interval costs far more than half the
        life because the radio wake cycle dominates.

        Solar-assisted sites should size panels for the worst month, not the
        annual average. A panel sized to the mean fails every winter and the
        failure looks like intermittent connectivity rather than a power problem,
        which sends engineers down the RF diagnostic path for nothing.

        Below -20 degrees, lithium thionyl chloride capacity drops sharply.
        Budget 40 per cent less usable capacity at Arctic sites."""),

    Document(
        "HW-04", "Antenna placement", "hardware", "2026-01-20",
        """Mount antennas vertically, with at least 30 centimetres of clearance
        from metal surfaces. An antenna flat against a metal panel loses
        approximately 12 dB, which typically halves usable range.

        Avoid mounting inside enclosures. An IP66 plastic enclosure costs 2 to
        3 dB; a metal one effectively blocks the link.

        Line of sight matters more than height beyond about six metres. Raising an
        antenna above the local obstruction is worth more than raising it
        further."""),

    # --------------------------------------------------------------- policies
    Document(
        "POL-01", "Hardware RMA policy", "policy", "2026-02-01",
        """Hardware is warranted for 36 months from shipment. An RMA requires a
        diagnostic code proving a hardware fault — for radios that is
        `helios-cli radio selftest` exit code 3.

        Field-recoverable conditions are not RMA-eligible. A bootloader loop, a
        full buffer, a drifted calibration and a misplaced antenna are all
        resolvable on site and will be rejected.

        Replacement units ship within two business days. The faulty unit must be
        returned within 30 days or the replacement is invoiced at list price."""),

    Document(
        "POL-02", "Data retention", "policy", "2026-02-01",
        """Raw telemetry is retained for 13 months, then downsampled to hourly
        averages and retained for a further five years.

        Exports are available over the full retention window, but exports
        spanning the downsampled period return hourly averages and are labelled
        as such in the response metadata.

        Deletion requests are honoured within 30 days and cascade to backups on
        the next backup rotation, which can take an additional 14 days."""),
]

BY_ID = {d.doc_id: d for d in DOCUMENTS}


def corpus_stats() -> dict:
    kinds: dict[str, int] = {}
    for d in DOCUMENTS:
        kinds[d.kind] = kinds.get(d.kind, 0) + 1
    return {"documents": len(DOCUMENTS), "by_kind": kinds}
