# Filler Word Handler Test Suite

## Overview

This test suite (`test_filler_word_handler.py`) comprehensively tests all scenarios from the challenge:

1. ✅ **User filler while agent speaks** → "uh", "hmm", "umm" → Ignored
2. ✅ **User real interruption** → "wait one second", "no not that one" → Allowed
3. ✅ **User filler while agent quiet** → "umm" → Registered
4. ✅ **Mixed filler and command** → "umm okay stop" → Allowed (contains command)
5. ✅ **Background murmur (low confidence)** → "hmm yeah" → Ignored if below threshold

## Running the Tests

### Prerequisites

1. Install the package in editable mode:
   ```bash
   pip install -e livekit-agents
   ```

2. Install test dependencies:
   ```bash
   pip install pytest pytest-asyncio
   ```

### Run All Tests

```bash
pytest tests/test_filler_word_handler.py -v
```

### Run Specific Test Classes

```bash
# Test filler word detection
pytest tests/test_filler_word_handler.py::TestFillerWordDetection -v

# Test scenario 1: Filler while agent speaking
pytest tests/test_filler_word_handler.py::TestScenario1_FillerWhileAgentSpeaking -v

# Test scenario 2: Real interruption
pytest tests/test_filler_word_handler.py::TestScenario2_RealInterruption -v

# Test scenario 3: Filler while agent quiet
pytest tests/test_filler_word_handler.py::TestScenario3_FillerWhileAgentQuiet -v

# Test scenario 4: Mixed filler and command
pytest tests/test_filler_word_handler.py::TestScenario4_MixedFillerAndCommand -v

# Test scenario 5: Low confidence transcripts
pytest tests/test_filler_word_handler.py::TestScenario5_LowConfidenceTranscript -v
```

### Run with Coverage

```bash
pytest tests/test_filler_word_handler.py --cov=livekit.agents.voice.filler_word_handler --cov-report=html
```

## Test Coverage

The test suite includes:

### Core Functionality Tests
- `TestFillerWordDetection`: Tests filler word and interrupt command detection logic
- `TestInterimTranscripts`: Tests interim transcript handling
- `TestStateManagement`: Tests agent speaking state transitions

### Scenario Tests (All Challenge Cases)
- `TestScenario1_FillerWhileAgentSpeaking`: Filler words ignored when agent speaking
- `TestScenario2_RealInterruption`: Interrupt commands always allowed
- `TestScenario3_FillerWhileAgentQuiet`: All transcripts registered when agent quiet
- `TestScenario4_MixedFillerAndCommand`: Mixed transcripts with commands allowed
- `TestScenario5_LowConfidenceTranscript`: Low confidence transcripts filtered

### Configuration Tests
- `TestConfiguration`: Tests custom configs and dynamic updates

### Edge Case Tests
- `TestEdgeCases`: Empty transcripts, whitespace, case sensitivity, word boundaries
- `TestComplexScenarios`: Rapid state changes, punctuation, complex combinations

## Test Structure

Each test:
1. Sets up the handler with appropriate configuration
2. Sets the agent speaking state
3. Creates a speech event
4. Calls the handler method
5. Asserts the expected behavior (forwarded or filtered)

## Example Test Output

```
tests/test_filler_word_handler.py::TestScenario1_FillerWhileAgentSpeaking::test_uh_while_speaking PASSED
tests/test_filler_word_handler.py::TestScenario1_FillerWhileAgentSpeaking::test_hmm_while_speaking PASSED
tests/test_filler_word_handler.py::TestScenario2_RealInterruption::test_wait_while_speaking PASSED
tests/test_filler_word_handler.py::TestScenario3_FillerWhileAgentQuiet::test_umm_while_quiet PASSED
tests/test_filler_word_handler.py::TestScenario4_MixedFillerAndCommand::test_umm_okay_stop_while_speaking PASSED
tests/test_filler_word_handler.py::TestScenario5_LowConfidenceTranscript::test_low_confidence_with_threshold PASSED
...
```

## Notes

- All tests use mocks to avoid external dependencies
- Tests are isolated and can run in any order
- No actual STT/VAD/LLM services required
- Fast execution (< 1 second for full suite)

