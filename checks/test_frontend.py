"""Native Streamlit flow tests; service responses are controlled, no live records."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from checks.test_accounts import PAYLOAD, prediction

APP = str(Path(__file__).resolve().parents[1] / 'frontend' / 'app.py')
USER = {'id': 1, 'full_name': 'Test Person', 'email': 'test@example.com'}
PROFILE = {'cnic': PAYLOAD['cnic'], 'total_visits': 1, 'total_approved': 1, 'total_rejected': 0,
           'last_decision': 'APPROVED', 'last_loan_amnt': 10000, 'last_interest_rate': 11.2,
           'first_seen': '2026-10-27', 'last_seen': '2026-10-27', 'last_visit': None}


def response(data, status=200):
    return SimpleNamespace(status_code=status, ok=status < 400, json=lambda: data)


def fake_request(method, url, **kwargs):
    if url.endswith('/auth/me'):
        return response(USER)
    if url.endswith('/health'):
        return response({'status': 'healthy', 'models': {'database': True, 'onnx': True}})
    if url.endswith(('/auth/login', '/auth/register')):
        return response({'token': 'test-token', 'user': USER})
    if '/predict/' in url:
        data = prediction(kwargs['json'], url.endswith('/existing'))
        return response({'prediction_id': 1, 'workflow': data['workflow'], **data['prediction_data']})
    if '/applicant/' in url:
        return response({'message': 'Deleted'} if method == 'DELETE' else PROFILE)
    raise AssertionError(f'Unexpected request {method} {url}')


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.mock = patch('requests.request', side_effect=fake_request)
        self.request = self.mock.start()
        self.addCleanup(self.mock.stop)
        self.ui = AppTest.from_file(APP).run(timeout=20)
        self.assertFalse(self.ui.exception)

    def button(self, label):
        matches = [b for b in self.ui.button if b.label == label]
        self.assertTrue(matches, f'{label!r} missing from {[(b.label, b.key) for b in self.ui.button]}')
        return matches[0]

    def text(self, label):
        return next(t for t in self.ui.text_input if t.label == label)

    def workspace(self):
        self.ui.session_state['token'] = 'test-token'
        self.ui.session_state['page'] = 'Workspace'
        self.ui.run()
        self.assertFalse(self.ui.exception)

    def test_public_landing_authentication_and_themes(self):
        self.request.assert_not_called()
        self.button('Get started →').click().run()
        self.assertFalse(self.ui.exception)
        self.text('Full name').set_value('Test Person')
        self.text('Email address').set_value('test@example.com')
        self.text('Password').set_value('password-123')
        self.text('Confirm password').set_value('mismatch')
        self.button('Create account →').click().run()
        self.assertTrue(any('do not match' in e.value for e in self.ui.error))
        self.assertEqual(self.text('Full name').value, 'Test Person')
        self.assertEqual(self.text('Email address').value, 'test@example.com')
        self.assertEqual(self.text('Password').value, 'password-123')
        self.assertEqual(self.text('Confirm password').value, 'mismatch')
        self.text('Full name').set_value('Test Person')
        self.text('Email address').set_value('test@example.com')
        self.text('Password').set_value('password-123')
        self.text('Confirm password').set_value('password-123')
        self.button('Create account →').click().run()
        self.assertFalse(self.ui.exception)
        self.assertEqual(self.ui.session_state['page'], 'Workspace')
        # Streamlit 1.29 AppTest omits earlier deltas after an in-form rerun.
        self.ui.run()
        self.assertEqual(len([b for b in self.ui.button if b.key and b.key.startswith('nav_')]), 4)
        self.button('◐').click().run()
        self.assertFalse(self.ui.exception)
        with patch('requests.post'):
            self.button('Log out').click().run()
        self.assertFalse(self.ui.exception)
        self.assertNotIn('token', self.ui.session_state)
        self.assertEqual(self.ui.session_state['page'], 'Log in')

    def test_both_prediction_forms_and_results(self):
        self.workspace()
        self.text('CNIC').set_value('invalid')
        self.button('Run assessment →').click().run()
        self.assertTrue(self.ui.error)
        self.text('CNIC').set_value('4210112345671')
        self.button('Run assessment →').click().run()
        self.assertFalse(self.ui.exception)
        self.assertTrue(any('Approved' in s.value for s in self.ui.success))
        self.button('Existing Loan').click().run()
        self.text('CNIC').set_value(PAYLOAD['cnic'])
        self.button('Run assessment →').click().run()
        self.assertFalse(self.ui.exception)
        self.assertTrue(any(m.label == 'Assigned rate' and m.value == '12.00%' for m in self.ui.metric))

    def test_profile_and_delete_confirmation(self):
        self.workspace()
        self.button('Applicant Profile').click().run()
        self.text('Applicant CNIC').set_value(PAYLOAD['cnic'])
        self.button('Find applicant').click().run()
        self.assertFalse(self.ui.exception)
        self.assertTrue(self.ui.metric)
        self.button('Delete Applicant').click().run()
        self.text('Applicant CNIC').set_value(PAYLOAD['cnic'])
        self.button('Review records').click().run()
        self.button('Permanently delete applicant').click().run()
        self.assertTrue(any('Type DELETE' in e.value for e in self.ui.error))
        self.assertFalse(any(c.args[0] == 'DELETE' for c in self.request.call_args_list))
        self.text('Type DELETE to confirm').set_value('DELETE')
        self.button('Permanently delete applicant').click().run()
        self.assertFalse(self.ui.exception)
        self.assertTrue(any(c.args[0] == 'DELETE' for c in self.request.call_args_list))
        self.assertTrue(any('were deleted' in i.value for i in self.ui.info))

    def test_expired_session_discards_cached_results(self):
        self.workspace()
        with patch('requests.request', return_value=response({'detail': 'Expired'}, 401)):
            self.ui.run()
        self.assertFalse(self.ui.exception)
        self.assertNotIn('token', self.ui.session_state)
        self.assertEqual(self.ui.session_state['page'], 'Log in')


if __name__ == '__main__':
    unittest.main()
