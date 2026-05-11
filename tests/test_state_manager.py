from __future__ import annotations

import time
import threading
import pytest

from src.proxy.state import StateManager


@pytest.fixture()
def sm():
    return StateManager(ttl_seconds=60)


# --------------------------------------------------------------------------- #
# set_mapping / get_mapping                                                    #
# --------------------------------------------------------------------------- #

def test_set_and_get_mapping(sm):
    sm.set_mapping("req-1", "[PERSON_1]", "Alice")
    assert sm.get_mapping("req-1") == {"[PERSON_1]": "Alice"}


def test_get_mapping_unknown_request_returns_empty(sm):
    assert sm.get_mapping("unknown") == {}


def test_multiple_tokens_same_request(sm):
    sm.set_mapping("req-2", "[PERSON_1]", "Bob")
    sm.set_mapping("req-2", "[EMAIL_ADDRESS_1]", "bob@example.com")
    mapping = sm.get_mapping("req-2")
    assert mapping["[PERSON_1]"] == "Bob"
    assert mapping["[EMAIL_ADDRESS_1]"] == "bob@example.com"


def test_get_mapping_returns_copy(sm):
    sm.set_mapping("req-3", "[PERSON_1]", "Carol")
    mapping = sm.get_mapping("req-3")
    mapping["[PERSON_1]"] = "tampered"
    # Original must be unchanged
    assert sm.get_mapping("req-3")["[PERSON_1]"] == "Carol"


# --------------------------------------------------------------------------- #
# get_original                                                                 #
# --------------------------------------------------------------------------- #

def test_get_original_existing(sm):
    sm.set_mapping("req-4", "[ORG_1]", "ACME Corp")
    assert sm.get_original("req-4", "[ORG_1]") == "ACME Corp"


def test_get_original_missing_token(sm):
    sm.set_mapping("req-5", "[PERSON_1]", "Dave")
    assert sm.get_original("req-5", "[EMAIL_ADDRESS_1]") is None


def test_get_original_missing_request(sm):
    assert sm.get_original("no-such-req", "[PERSON_1]") is None


# --------------------------------------------------------------------------- #
# delete_request                                                               #
# --------------------------------------------------------------------------- #

def test_delete_request_removes_mapping(sm):
    sm.set_mapping("req-6", "[PERSON_1]", "Eve")
    sm.delete_request("req-6")
    assert sm.get_mapping("req-6") == {}


def test_delete_request_idempotent(sm):
    # Deleting a non-existent request must not raise
    sm.delete_request("ghost")


# --------------------------------------------------------------------------- #
# TTL cleanup                                                                  #
# --------------------------------------------------------------------------- #

def test_cleanup_removes_expired_entries():
    sm = StateManager(ttl_seconds=0)  # Expires immediately
    sm.set_mapping("old-req", "[PERSON_1]", "Frank")
    time.sleep(0.01)  # Ensure TTL passes
    sm.cleanup()
    assert sm.get_mapping("old-req") == {}


def test_cleanup_keeps_fresh_entries():
    sm = StateManager(ttl_seconds=3600)
    sm.set_mapping("fresh-req", "[PERSON_1]", "Grace")
    sm.cleanup()
    assert sm.get_mapping("fresh-req") == {"[PERSON_1]": "Grace"}


# --------------------------------------------------------------------------- #
# Thread safety                                                                #
# --------------------------------------------------------------------------- #

def test_concurrent_writes_are_safe(sm):
    errors = []

    def write(i):
        try:
            sm.set_mapping("concurrent", f"[TOKEN_{i}]", f"value-{i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(sm.get_mapping("concurrent")) == 50
