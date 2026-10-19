# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-10-19

First release.

### Added
- FastAPI service with a single `POST /analyze` endpoint for stereo call recordings.
- Speaker separation by channel: left is the client, right is the operator.
- Silero VAD segmentation with configurable pause and padding.
- Per-utterance emotion probabilities and the dominant emotion.
- Splitting of fragments longer than 25 seconds, the GigaAM input limit.
- CPU and CUDA inference, model and device selection through environment variables.
