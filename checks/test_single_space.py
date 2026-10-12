"""Offline checks for readiness and the combined-service launcher."""
import subprocess
import unittest
from unittest.mock import Mock, patch

import requests
from frontend.backend_client import wait_for_api
import start_space


class ReadinessTests(unittest.TestCase):
    def response(self, status_code=200, **data):
        response = Mock(status_code=status_code)
        response.json.return_value = data
        return response

    def test_waits_through_connection_errors_unhealthy_and_invalid_json(self):
        invalid = self.response()
        invalid.json.side_effect = ValueError('not JSON')
        responses = [requests.ConnectionError(), self.response(503), invalid,
                     self.response(status='unhealthy', models={'database': False}),
                     self.response(status='healthy', models={'database': True, 'cls_onnx': True})]
        with patch('frontend.backend_client.requests.get', side_effect=responses) as get, \
             patch('frontend.backend_client.time.sleep'):
            self.assertTrue(wait_for_api('http://localhost:8000/'))
        self.assertEqual(get.call_count, 5)
        self.assertEqual(get.call_args.args[0], 'http://localhost:8000/health')

    def test_http_200_with_missing_or_false_models_does_not_unlock(self):
        for models in [{}, {'database': False}, {'database': 'true'}]:
            with self.subTest(models=models), \
                 patch('frontend.backend_client.requests.get', return_value=self.response(status='healthy', models=models)), \
                 patch('frontend.backend_client.time.monotonic', side_effect=[0, 0, 1]), \
                 patch('frontend.backend_client.time.sleep'):
                self.assertFalse(wait_for_api('http://localhost:8000', max_wait=1))

    def test_timeout_is_bounded_and_zero_wait_makes_no_request(self):
        with patch('frontend.backend_client.requests.get') as get:
            self.assertFalse(wait_for_api('http://localhost:8000', max_wait=0))
        get.assert_not_called()
        with patch('frontend.backend_client.requests.get', side_effect=requests.Timeout()) as get, \
             patch('frontend.backend_client.time.monotonic', side_effect=[0, 0, 1]):
            self.assertFalse(wait_for_api('http://localhost:8000', max_wait=1))
        self.assertEqual(get.call_args.kwargs['timeout'], 1)


class LauncherTests(unittest.TestCase):
    def test_child_failure_stops_other_service_and_forces_internal_url(self):
        api = Mock()
        api.poll.return_value = 7
        ui = Mock()
        ui.poll.return_value = None
        with patch('start_space.subprocess.Popen', side_effect=[api, ui]) as popen, \
             patch('start_space.signal.signal'), \
             patch('start_space.sys.argv', ['start_space.py']), \
             patch.dict('os.environ', {'API_URL': 'https://old-api.hf.space'}):
            self.assertEqual(start_space.main(), 7)
        ui.terminate.assert_called_once()
        self.assertEqual(popen.call_args.kwargs['env']['API_URL'], 'http://127.0.0.1:8000')

    def test_shutdown_kills_a_child_that_will_not_terminate(self):
        child = Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('service', 10), 0]
        start_space.stop_children([child])
        child.terminate.assert_called_once()
        child.kill.assert_called_once()


if __name__ == '__main__':
    unittest.main()

