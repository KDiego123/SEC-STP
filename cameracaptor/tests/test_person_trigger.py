import unittest

from src.person_trigger import PersonArrivalTrigger


class PersonArrivalTriggerTests(unittest.TestCase):
    def test_alert_only_after_observed_idle_and_two_confirmations(self):
        trigger = PersonArrivalTrigger(12.0, 2, started_at=0.0)
        self.assertFalse(trigger.observe(False, 11.0))
        self.assertFalse(trigger.observe(True, 12.5))
        self.assertTrue(trigger.observe(True, 13.0))
        self.assertFalse(trigger.observe(True, 30.0))

    def test_short_absence_does_not_repeat(self):
        trigger = PersonArrivalTrigger(12.0, 2, started_at=0.0)
        trigger.observe(True, 13.0)
        trigger.observe(True, 13.5)
        trigger.observe(False, 20.0)
        self.assertFalse(trigger.observe(True, 25.0))
        self.assertFalse(trigger.observe(True, 25.5))
        trigger.observe(False, 26.0)
        self.assertFalse(trigger.observe(True, 39.0))
        self.assertTrue(trigger.observe(True, 39.5))

    def test_startup_presence_does_not_count_as_idle(self):
        trigger = PersonArrivalTrigger(12.0, 2, started_at=0.0)
        self.assertFalse(trigger.observe(True, 1.0))
        self.assertFalse(trigger.observe(True, 2.0))
        trigger.observe(False, 3.0)
        self.assertFalse(trigger.observe(True, 15.0))
        self.assertTrue(trigger.observe(True, 15.5))

    def test_reset_requires_new_idle_window(self):
        trigger = PersonArrivalTrigger(12.0, 2, started_at=0.0)
        trigger.reset(20.0)
        self.assertFalse(trigger.observe(True, 25.0))
        self.assertFalse(trigger.observe(True, 25.5))

    def test_unobserved_startup_time_does_not_arm(self):
        trigger = PersonArrivalTrigger(12.0, 2)
        self.assertFalse(trigger.observe(True, 1000.0))
        self.assertFalse(trigger.observe(True, 1000.5))
        trigger.observe(False, 1001.0)
        self.assertFalse(trigger.observe(True, 1014.0))
        self.assertTrue(trigger.observe(True, 1014.5))


if __name__ == "__main__":
    unittest.main()
