"""Agrios perception cell: RTSP ingest, detection, tracking, pose, behavioural
analytics, and the evaluation harness that every one of those is measured by.

Phase 1 ships the harness first, before any model: the metric calculators
(vision.eval.metrics), the manifest schema (vision.eval.manifest), seeded
synthetic streams (vision.synth), the temporal alert state machine
(vision.alerts) and a deliberately simple classical reference pipeline
(vision.reference) that exists only so the gate has something end-to-end to
run -- and so its negative controls have something to fail.
"""
