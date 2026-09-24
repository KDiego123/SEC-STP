import unittest

from src.person_trigger import (
    ENTRY_ALERT,
    PersonAlertSequence,
    PersonArrivalTrigger,
)


class PersonArrivalTriggerTests(unittest.TestCase):
    def test_alert_immediately_after_sustained_presence(self):
        trigger = PersonArrivalTrigger(12.0, 1.5, started_at=0.0)
        self.assertFalse(trigger.observe(False, 11.0))
        self.assertFalse(trigger.observe(True, 12.5))
        self.assertFalse(trigger.observe(True, 13.99))
        self.assertTrue(trigger.observe(True, 14.0))
        self.assertFalse(trigger.observe(True, 30.0))

    def test_short_absence_does_not_repeat(self):
        trigger = PersonArrivalTrigger(12.0, 1.5, started_at=0.0)
        trigger.observe(True, 13.0)
        trigger.observe(True, 14.5)
        trigger.observe(False, 20.0)
        self.assertFalse(trigger.observe(True, 25.0))
        self.assertFalse(trigger.observe(True, 30.0))
        trigger.observe(False, 31.0)
        self.assertFalse(trigger.observe(True, 44.0))
        self.assertFalse(trigger.observe(True, 45.49))
        self.assertTrue(trigger.observe(True, 45.5))

    def test_startup_presence_does_not_count_as_idle(self):
        trigger = PersonArrivalTrigger(12.0, 1.5, started_at=0.0)
        self.assertFalse(trigger.observe(True, 1.0))
        self.assertFalse(trigger.observe(True, 2.0))
        trigger.observe(False, 3.0)
        self.assertFalse(trigger.observe(True, 15.0))
        self.assertFalse(trigger.observe(True, 16.49))
        self.assertTrue(trigger.observe(True, 16.5))

    def test_reset_requires_new_idle_window(self):
        trigger = PersonArrivalTrigger(12.0, 1.5, started_at=0.0)
        trigger.reset(20.0)
        self.assertFalse(trigger.observe(True, 25.0))
        self.assertFalse(trigger.observe(True, 30.0))

    def test_unobserved_startup_time_does_not_arm(self):
        trigger = PersonArrivalTrigger(12.0, 1.5)
        self.assertFalse(trigger.observe(True, 1000.0))
        self.assertFalse(trigger.observe(True, 1000.5))
        trigger.observe(False, 1001.0)
        self.assertFalse(trigger.observe(True, 1014.0))
        self.assertFalse(trigger.observe(True, 1015.49))
        self.assertTrue(trigger.observe(True, 1015.5))

    def test_idle_elapsed_ignores_brief_detection_gap(self):
        trigger = PersonArrivalTrigger(12.0, 1.5)
        trigger.observe(False, 10.0)
        self.assertFalse(trigger.idle_elapsed(21.9))
        self.assertTrue(trigger.idle_elapsed(22.0))
        trigger.observe(True, 22.1)
        self.assertFalse(trigger.idle_elapsed(30.0))

    def test_detection_gap_restarts_presence_validation(self):
        trigger = PersonArrivalTrigger(12.0, 1.5, started_at=0.0)
        self.assertFalse(trigger.observe(True, 13.0))
        self.assertFalse(trigger.observe(True, 14.4))
        self.assertFalse(trigger.observe(False, 14.5))
        self.assertFalse(trigger.observe(True, 27.0))
        self.assertFalse(trigger.observe(True, 28.49))
        self.assertTrue(trigger.observe(True, 28.5))

    def test_strong_detection_can_use_shorter_confirmation(self):
        trigger = PersonArrivalTrigger(12.0, 1.5, started_at=0.0)
        self.assertFalse(trigger.observe(True, 13.0, confirmation_seconds=0.6))
        self.assertFalse(trigger.observe(True, 13.59, confirmation_seconds=0.6))
        self.assertTrue(trigger.observe(True, 13.61, confirmation_seconds=0.6))

    def test_invalid_dynamic_confirmation_is_rejected(self):
        trigger = PersonArrivalTrigger(12.0, 1.5, started_at=0.0)
        with self.assertRaises(ValueError):
            trigger.observe(True, 13.0, confirmation_seconds=0.0)


class PersonAlertSequenceTests(unittest.TestCase):
    def test_inactive_sequence_does_not_announce(self):
        sequence = PersonAlertSequence()
        self.assertEqual(sequence.announcements(("Brayan",)), ())

    def test_entry_is_announced_before_known_identity(self):
        sequence = PersonAlertSequence()
        sequence.arm()
        messages = sequence.announcements(("Brayan",))
        self.assertEqual([message.text for message in messages], [
            ENTRY_ALERT,
            "Bienvenido Brayan",
        ])

    def test_late_identity_creates_second_announcement(self):
        sequence = PersonAlertSequence()
        sequence.arm()
        self.assertEqual(sequence.announcements(())[0].text, ENTRY_ALERT)
        messages = sequence.announcements(("Brayan",))
        self.assertEqual([message.text for message in messages], ["Bienvenido Brayan"])

    def test_identity_is_not_repeated_during_same_arrival(self):
        sequence = PersonAlertSequence()
        sequence.arm()
        sequence.announcements(("Brayan",))
        self.assertEqual(sequence.announcements(("Brayan",)), ())
        sequence.reset()
        sequence.arm()
        self.assertEqual(len(sequence.announcements(("Brayan",))), 2)


if __name__ == "__main__":
    unittest.main()
