from __future__ import annotations
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pipeline import Event


TAU_BENCH_SPEC = {
    "account":   {"X_vars": ["auth_status", "account_tier", "email_verified", "address_on_file"],
                  "A_vars": ["authenticate_user", "verify_zip", "update_address"],
                  "I_out": ["account_authenticated", "address_updated"], "I_in": []},
    "order":     {"X_vars": ["order_status", "cart_items", "order_total", "cart_locked"],
                  "A_vars": ["create_order", "modify_pending_order", "confirm_order", "cancel_order"],
                  "I_out": ["order_placed", "order_confirmed", "order_cancelled", "order_delivered"],
                  "I_in": ["account_authenticated", "payment_captured", "inventory_reserved",
                           "shipment_completed", "refund_issued"]},
    "payment":   {"X_vars": ["payment_status", "gift_card_balance", "auth_amount", "capture_amount"],
                  "A_vars": ["authorize_payment", "capture_payment", "refund_payment"],
                  "I_out": ["payment_authorized", "payment_captured", "payment_failed", "payment_refunded"],
                  "I_in": ["order_placed", "refund_issued"]},
    "inventory": {"X_vars": ["available_qty", "reserved_qty", "restock_eta"],
                  "A_vars": ["check_availability", "reserve", "release_reservation"],
                  "I_out": ["inventory_reserved", "inventory_oos", "inventory_released"],
                  "I_in": ["order_placed", "order_cancelled", "order_delivered"]},
    "shipment":  {"X_vars": ["shipment_status", "carrier", "tracking_id", "est_delivery_days"],
                  "A_vars": ["create_label", "pickup_request", "mark_delivered"],
                  "I_out": ["shipment_labeled", "shipment_in_transit", "shipment_completed"],
                  "I_in": ["payment_captured", "inventory_reserved"]},
    "refund":    {"X_vars": ["refund_eligibility", "refund_method", "refund_amount", "claim_status"],
                  "A_vars": ["submit_refund_claim", "approve_claim", "issue_refund"],
                  "I_out": ["refund_eligibility_updated", "refund_issued"],
                  "I_in": ["shipment_completed", "payment_refunded"]},
}

TAU_BENCH_TEMPLATE = [
    ("account",   "account_authenticated", "order"),
    ("order",     "order_placed",          "payment"),
    ("order",     "order_placed",          "inventory"),
    ("inventory", "inventory_reserved",    "order"),
    ("payment",   "payment_captured",      "order"),
    ("order",     "order_confirmed",       "shipment"),
    ("inventory", "inventory_reserved",    "shipment"),
    ("shipment",  "shipment_completed",    "order"),
    ("shipment",  "shipment_completed",    "refund"),
    ("refund",    "refund_issued",         "payment"),
    ("payment",   "payment_refunded",      "order"),
    ("order",     "order_cancelled",       "inventory"),
]

ALFWORLD_SPEC = {
    "navigation":         {"X_vars": ["agent_loc", "agent_facing"],
                           "A_vars": ["goto", "look"],
                           "I_out": ["agent_at", "reachable_updated"], "I_in": []},
    "container_access":   {"X_vars": ["container_open", "visible_inside"],
                           "A_vars": ["open", "close"],
                           "I_out": ["container_opened", "object_reachable"], "I_in": ["agent_at"]},
    "object_manipulation":{"X_vars": ["holding", "object_loc"],
                           "A_vars": ["pick", "place", "examine"],
                           "I_out": ["object_picked", "object_placed", "object_inserted_appliance"],
                           "I_in": ["object_reachable", "agent_at"]},
    "appliance":          {"X_vars": ["appliance_state", "appliance_timer"],
                           "A_vars": ["toggle", "use"],
                           "I_out": ["appliance_on", "effect_applied"],
                           "I_in": ["object_inserted_appliance", "agent_at"]},
    "object_property":    {"X_vars": ["is_clean", "is_heated", "is_cooled", "is_cooked", "is_sliced"],
                           "A_vars": ["slice"],
                           "I_out": ["property_changed"],
                           "I_in": ["effect_applied", "object_placed"]},
    "task_monitor":       {"X_vars": ["goal_predicates", "satisfied_predicates", "task_complete"],
                           "A_vars": [],
                           "I_out": ["goal_satisfied", "task_completed"],
                           "I_in": ["property_changed", "object_placed", "agent_at"]},
}

ALFWORLD_TEMPLATE = [
    ("navigation",          "agent_at",                "container_access"),
    ("container_access",    "container_opened",        "object_manipulation"),
    ("navigation",          "agent_at",                "object_manipulation"),
    ("object_manipulation", "object_inserted_appliance","appliance"),
    ("appliance",           "effect_applied",          "object_property"),
    ("object_manipulation", "object_placed",           "object_property"),
    ("object_property",     "property_changed",        "task_monitor"),
    ("object_manipulation", "object_placed",           "task_monitor"),
]

ANDROIDWORLD_SPEC = {
    "permissions_settings": {"X_vars": ["permission_granted", "airplane_mode", "wifi_connected"],
                             "A_vars": ["grant_permission", "revoke_permission", "toggle_airplane"],
                             "I_out": ["permission_granted", "setting_changed", "network_state_changed"],
                             "I_in": []},
    "contacts":             {"X_vars": ["contact_db", "last_modified_id"],
                             "A_vars": ["add_contact", "edit_contact", "delete_contact"],
                             "I_out": ["contact_added", "contact_updated", "contact_deleted"],
                             "I_in": ["permission_granted"]},
    "messaging":            {"X_vars": ["outbox", "inbox", "recipient_resolvable"],
                             "A_vars": ["send_sms", "reply_sms"],
                             "I_out": ["sms_sent", "sms_send_failed"],
                             "I_in": ["permission_granted", "contact_added", "network_state_changed"]},
    "files":                {"X_vars": ["fs_tree", "available_storage_mb", "last_written_path"],
                             "A_vars": ["write_file", "delete_file", "take_photo"],
                             "I_out": ["file_created", "file_deleted", "media_added"],
                             "I_in": ["permission_granted"]},
    "calendar":             {"X_vars": ["event_list", "reminder_list"],
                             "A_vars": ["add_event", "add_reminder", "delete_event"],
                             "I_out": ["event_created", "reminder_scheduled"],
                             "I_in": ["permission_granted", "contact_added"]},
    "notifications":        {"X_vars": ["notif_queue", "do_not_disturb"],
                             "A_vars": ["dismiss", "tap"],
                             "I_out": ["notif_dismissed", "notif_tapped"],
                             "I_in": ["sms_sent", "reminder_scheduled", "file_created", "media_added"]},
}

ANDROIDWORLD_TEMPLATE = [
    ("permissions_settings", "permission_granted", "contacts"),
    ("permissions_settings", "permission_granted", "messaging"),
    ("permissions_settings", "permission_granted", "files"),
    ("permissions_settings", "permission_granted", "calendar"),
    ("contacts",             "contact_added",      "messaging"),
    ("contacts",             "contact_added",      "calendar"),
    ("permissions_settings", "network_state_changed","messaging"),
    ("messaging",            "sms_sent",           "notifications"),
    ("files",                "media_added",        "notifications"),
    ("calendar",             "reminder_scheduled", "notifications"),
]


BENCHMARKS = {
    "tau_bench":    (TAU_BENCH_SPEC,    TAU_BENCH_TEMPLATE,    12),
    "alfworld":     (ALFWORLD_SPEC,     ALFWORLD_TEMPLATE,     20),
    "androidworld": (ANDROIDWORLD_SPEC, ANDROIDWORLD_TEMPLATE, 25),
}


@dataclass
class Task:
    task_id: str
    benchmark: str
    events: List[Event]
    target_event: Event
    ground_truth_edges: List[tuple]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "events": [asdict(e) for e in self.events],
            "target_event": asdict(self.target_event),
            "ground_truth_edges": self.ground_truth_edges,
        }


def _sample_intervention_id(rng: random.Random) -> str:
    return f"int_{rng.randint(0, 100000):06d}"


def synth_task(benchmark: str, task_idx: int, seed: int = 0) -> Task:
    spec, template, target_events = BENCHMARKS[benchmark]
    rng = random.Random(seed * 1000 + task_idx)
    events: List[Event] = []
    t = 0
    for round_idx in range((target_events // max(len(template), 1)) + 1):
        for (src_mod, src_evt, tgt_mod) in template:
            tgt_evt = src_evt
            intervention = _sample_intervention_id(rng) if rng.random() < 0.30 else None
            events.append(Event(src_mod, src_evt, t, payload={"round": round_idx},
                                intervention_id=intervention))
            t += rng.randint(1, 3)
            events.append(Event(tgt_mod, tgt_evt, t, payload={"round": round_idx}))
            t += rng.randint(1, 3)
            if len(events) >= target_events + 1:
                break
        if len(events) >= target_events + 1:
            break

    target = events[-1]
    prefix = events[:-1]
    gt_edges = [(s, se, t_, se) for (s, se, t_) in template]
    return Task(task_id=f"{benchmark}_task_{task_idx:03d}",
                benchmark=benchmark, events=prefix,
                target_event=target, ground_truth_edges=gt_edges)


def synth_task_pool(benchmark: str, n_tasks: int, seed: int = 0) -> List[Task]:
    return [synth_task(benchmark, i, seed) for i in range(n_tasks)]


if __name__ == "__main__":
    for b in BENCHMARKS:
        tasks = synth_task_pool(b, 2)
        print(f"=== {b} (2 sample tasks) ===")
        for t in tasks:
            print(f"  {t.task_id}: {len(t.events)} prefix events, "
                  f"target={t.target_event.module_id}.{t.target_event.event_type}")
