"""Regression test: every sidebar item must open with ONE click (previously every 2nd click was ignored)."""
import os, unittest
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


class TestSingleClickNavigation(unittest.TestCase):
    def test_each_menu_item_navigates_on_first_click(self):
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        options = list(at.sidebar.radio[0].options)
        # Visit in an order that includes repeats and back-and-forth jumps
        for label in options + options[::-1] + ["⚙️ Settings", "🏠 Dashboard", "⚙️ Settings"]:
            at.sidebar.radio[0].set_value(label).run()
            self.assertEqual(at.session_state["active_page"], label.split(" ", 1)[-1], label)
            self.assertFalse(at.exception)

    def test_in_page_button_updates_menu(self):
        at = AppTest.from_file(APP, default_timeout=60)
        at.run()
        next(b for b in at.button if "Open Summarizer" in b.label).click().run()
        self.assertEqual(at.session_state["active_page"], "Summarize")
        self.assertEqual(at.sidebar.radio[0].value, "📝 Summarize")


if __name__ == "__main__":
    unittest.main()
